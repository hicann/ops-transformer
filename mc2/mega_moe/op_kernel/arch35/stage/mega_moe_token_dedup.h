/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mega_moe_token_dedup.h
 * \brief MTE 通信去重：rowDesc prepass 建表 + 去重版 Token Dispatch。
 *
 * 背景：MTE dispatch 为 pull 模式，接收卡按源 tokenIndex 从远端窗口拉取记录。topK 路由下同一
 * 源 token 常被本卡多个专家选中，基线会把同一远端记录跨片拉 K 次。去重后每个 (srcRank, token)
 * 只拉一次，由组首行从 UB 直接扇出写到组内全部目的行（数据/scale/metaInfo 三区），数值 bit 恒等。
 */

#ifndef MEGA_MOE_TOKEN_DEDUP_H
#define MEGA_MOE_TOKEN_DEDUP_H

#include "../common/mega_moe_utils.h"

namespace MegaMoeImpl {

using namespace AscendC;

// ============================== 去重 prepass（建表） ==============================
// 干什么：在 dispatch 搬数据之前，先把「哪些行是重复的、谁跟谁一组」算出来，写成每行
// 一条的描述表（rowDesc）。同一 token 发到同一张卡的多行为一组：组里第一行标 FIRST
// （真正拉数据并扇出给全组），其余行标 SKIP（什么都不做），组内最大专家那行标 LAST
// （combine 合并由它驱动）。不在任何组里的行标 PLAIN，走基线路径。
// dispatch 去重与 combine 去重都按这张表办事，表只建一次。
// =================================================================================

/*
 * 去重 UB 暂存从 dispatch 固定区尾（scratch.dedupUbBaseAddr，512B 对齐）动态起址。
 * host 在 dedupMode!=0 时已从 dispatch 自适应预算预留 DEDUP_DISPATCH_UB_RESERVE_BYTES（含对齐余量），
 * 任意合法规格下不越 UB。prepass 与 dispatch 批内暂存分时复用同一段（prepass 先于任何 dispatch）。
 */
// prepass 布局（相对 dedupUbBaseAddr 的偏移）：route 装载 + 逐 token 计数 + desc 拼装 + 成员表。
constexpr uint32_t DEDUP_DESC_ASM_UB_BYTES = 2U * 1024U;
// dispatch 去重批内布局（相对 dedupUbBaseAddr）：desc 装载区 + 扇出 meta 拼装区 + 活跃条目索引区。
constexpr uint32_t DEDUP_DESC_BATCH_UB_BYTES = 48U * 1024U;
constexpr uint32_t DEDUP_FAN_META_UB_BYTES = 8U * 1024U; // bufferCount(≤6) × topK(≤32) × 32B 上界
constexpr uint32_t DEDUP_ACTIVE_LIST_UB_BYTES = 8U * 1024U;

// prepass 逐 token 打包状态字：低 8 位幸存成员数（<clamp）、次 8 位全量出现数、高 8 位全量最小 topkIdx。
// minK 必须按全量成员算：clamp 裁掉最小 k 成员时，发送槽仍要与归属卡按完整 topkIds 推导的
// canonical 槽一致（退化语义与基线截断对齐——被裁行的贡献丢失、幸存行保留）。
constexpr int32_t DEDUP_PACK_INIT = 0x00FF0000;

// ---------- 去重结构体（集中定义；各字段用途见行内注释） ----------

struct DedupConfig {
    GM_ADDR rowDescPtr = nullptr; // workspace rowDesc 表基址
    GM_ADDR prepassReadyPtr = nullptr; // prepass 完成标记区（槽0=每核done计数，槽1+item=项ready；随 launch 清零）
    int32_t dedupMode = 0;           // 同 tilingData->dedupMode（0=全关 1=仅combine 2=仅dispatch 3=双开）
    uint32_t rowDescStrideInt32 = 0; // 每行描述符 int32 数（CalcDedupRowDescStrideInt32）
    uint32_t prepassChunkTokens = 1; // 工作项 token 切片宽（CalcDedupPrepassChunkTokens 单源）
    uint32_t prepassChunkCount = 1;  // 每 srcRank 的 chunk 数 = ceil(numTokens/chunkTokens)
};

// 本工作项的 token 窗口：本段负责的源 token 起点，以及 route 值域上下界 [targetLo, targetHi)。
struct DedupChunkWindow {
    int32_t tokenLo = 0;
    int32_t targetLo = 0;
    int32_t targetHi = 0;
};

// prepass 工人上下文：UB 视图、GM 视图与跨工作项常量/状态，一次构造、各流水段共用。
template <typename TopkIndexType>
struct DedupPrepassWorker {
    static constexpr int32_t ROUTE_HALF_ITEMS =
        static_cast<int32_t>(DEDUP_ROUTE_LOAD_UB_BYTES / 2U / sizeof(TopkIndexType));
    static constexpr uint32_t DESC_ASM_HALF_INT32 = DEDUP_DESC_ASM_UB_BYTES / 2U / sizeof(int32_t);

    LocalTensor<TopkIndexType> routeLoadTensor; // route 槽装载区（两半双缓冲）
    LocalTensor<int32_t> countTensor;           // 逐 token 打包状态字（DEDUP_PACK_INIT 语义）
    LocalTensor<int32_t> memberTensor;          // 逐 token 成员三元组 (row, topkIndex, flagSlotOffset)
    LocalTensor<int32_t> descAsmTensor;         // desc 拼装区（两半乒乓）
    GlobalTensor<int32_t> rowDescGm;            // workspace rowDesc 表
    uint32_t topK = 0;
    uint32_t strideInt32 = 0;
    uint32_t maxOutputSize = 0;
    int32_t dispatchFlagSlotCountPerExpert = 0;
    uint32_t descHalf = 0; // desc 乒乓半区，跨工作项持续轮转
};

// 槽间双缓冲的在途槽：装载已发起、扫描尚未执行。expertIdx < 0 表示无在途。
struct DedupPendingSlot {
    int32_t expertIdx = -1;
    int32_t rowBase = 0;
    int32_t slotCount = 0;
    int32_t expertRowBegin = 0;
    uint32_t half = 0;
};

/*
 * prepass 工作项的 token 切片宽度（prepass 建表与 dispatch 按需等待两侧必须用同一公式）。
 * 三重约束：①UB 容量硬上限（member/count/route 三区装得下）②负载目标 = 工作项数 ≈ 2×AIV1 数
 * ③chunkCount ≤ DEDUP_PREPASS_MAX_CHUNKS（flag 区按此上界分配）。
 */
__aicore__ inline uint32_t CalcDedupPrepassChunkTokens(uint32_t numTokens, uint32_t worldSize, uint32_t topK,
                                                       uint32_t aivJobs)
{
    uint32_t chunk = DEDUP_MEMBER_UB_BYTES / (topK * 3U * static_cast<uint32_t>(sizeof(int32_t)));
    chunk = chunk > 512U ? 512U : chunk;
    uint32_t countCapacity = DEDUP_COUNT_UB_BYTES / static_cast<uint32_t>(sizeof(int32_t));
    chunk = chunk > countCapacity ? countCapacity : chunk;
    uint32_t routeCapacity = DEDUP_ROUTE_LOAD_UB_BYTES / static_cast<uint32_t>(sizeof(int32_t));
    chunk = chunk > routeCapacity ? routeCapacity : chunk;
    uint32_t targetItems = aivJobs * 2U;
    if (targetItems > 0U) {
        uint32_t balanced = (numTokens * worldSize + targetItems - 1U) / targetItems;
        chunk = balanced < chunk ? balanced : chunk;
    }
    uint32_t floorChunk = (numTokens + DEDUP_PREPASS_MAX_CHUNKS - 1U) / DEDUP_PREPASS_MAX_CHUNKS;
    chunk = chunk < floorChunk ? floorChunk : chunk;
    return chunk == 0U ? 1U : chunk;
}

__aicore__ inline DedupConfig CreateDedupConfig(const Params &params)
{
    DedupConfig config;
    config.dedupMode = params.tilingData->dedupMode;
    if (config.dedupMode == 0) {
        return config;
    }
    config.rowDescPtr = params.workspaceInfo.dedupRowDescPtr;
    config.prepassReadyPtr = params.workspaceInfo.dedupPrepassReadyPtr;
    config.rowDescStrideInt32 =
        static_cast<uint32_t>(CalcDedupRowDescStrideInt32(static_cast<int64_t>(params.tilingData->topK)));
    const uint32_t numTokens = params.tilingData->numMaxTokensPerRank;
    config.prepassChunkTokens = CalcDedupPrepassChunkTokens(numTokens, params.tilingData->epWorldSize,
                                                            params.tilingData->topK, params.tilingData->aicNum);
    config.prepassChunkCount = (numTokens + config.prepassChunkTokens - 1U) / config.prepassChunkTokens;
    return config;
}

// route 双缓冲半区事件：半区 0 用 SYNC_EVENT_ID4、半区 1 用 SYNC_EVENT_ID6。
__aicore__ inline TEventID DedupRouteHalfEvent(uint32_t half)
{
    return static_cast<TEventID>(half == 0U ? SYNC_EVENT_ID4 : SYNC_EVENT_ID6);
}

// desc 落盘乒乓半区事件：半区 0 用 SYNC_EVENT_ID3、半区 1 用 SYNC_EVENT_ID5。
__aicore__ inline TEventID DedupDescHalfEvent(uint32_t half)
{
    return static_cast<TEventID>(half == 0U ? SYNC_EVENT_ID3 : SYNC_EVENT_ID5);
}

// 在去重 UB 暂存段上建立 prepass 工人视图。route 槽元素类型与基线 dispatch 同源
// （int16/int32 由 UseInt16TopkIndex 选定）；UB 字节预算不变，条数随元素宽度变。
template <typename TopkIndexType, typename DispatchScratch>
__aicore__ inline DedupPrepassWorker<TopkIndexType> CreateDedupPrepassWorker(const DedupConfig &dedup,
                                                                             const MoeSyncWorkspaceLayout &syncLayout,
                                                                             const Params &params,
                                                                             DispatchScratch &scratch)
{
    DedupPrepassWorker<TopkIndexType> worker;
    const uint32_t routeLoadAddr = scratch.dedupUbBaseAddr;
    const uint32_t countAddr = routeLoadAddr + DEDUP_ROUTE_LOAD_UB_BYTES;
    const uint32_t descAsmAddr = countAddr + DEDUP_COUNT_UB_BYTES;
    const uint32_t memberAddr = descAsmAddr + DEDUP_DESC_ASM_UB_BYTES;
    worker.routeLoadTensor = LocalTensor<TopkIndexType>(TPosition::VECCALC, routeLoadAddr,
                                                        DEDUP_ROUTE_LOAD_UB_BYTES / sizeof(TopkIndexType));
    worker.countTensor = LocalTensor<int32_t>(TPosition::VECCALC, countAddr, DEDUP_COUNT_UB_BYTES / sizeof(int32_t));
    worker.descAsmTensor =
        LocalTensor<int32_t>(TPosition::VECCALC, descAsmAddr, DEDUP_DESC_ASM_UB_BYTES / sizeof(int32_t));
    worker.memberTensor = LocalTensor<int32_t>(TPosition::VECCALC, memberAddr, DEDUP_MEMBER_UB_BYTES / sizeof(int32_t));
    worker.rowDescGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(dedup.rowDescPtr));
    worker.topK = params.tilingData->topK;
    worker.strideInt32 = dedup.rowDescStrideInt32;
    worker.maxOutputSize = params.tilingData->maxOutputSize;
    worker.dispatchFlagSlotCountPerExpert = syncLayout.dispatchFlagSlotCountPerExpert;
    return worker;
}

// 对已装载进 UB 半区的一个 route 槽段做判界扫描，更新 count/member 表。
// 槽内按 topkIndex 升序，故低于窗口下界 continue、达到上界即 break。
template <typename TopkIndexType>
__aicore__ inline void DedupPrepassScanSlot(DedupPrepassWorker<TopkIndexType> &worker,
                                            LocalTensor<TopkIndexType> &routeSeg, const DedupChunkWindow &window,
                                            int32_t segCount, int32_t rowBase, int32_t segBase, uint32_t expertIdx,
                                            int32_t expertRowBegin)
{
    for (int32_t entryIdx = 0; entryIdx < segCount; ++entryIdx) {
        const int32_t topkIndex = static_cast<int32_t>(routeSeg.GetValue(entryIdx));
        if (topkIndex < window.targetLo) {
            continue;
        }
        if (topkIndex >= window.targetHi) {
            break;
        }
        const int32_t row = rowBase + segBase + entryIdx;
        const int32_t localToken = topkIndex / static_cast<int32_t>(worker.topK) - window.tokenLo;
        const int32_t k = topkIndex % static_cast<int32_t>(worker.topK);
        int32_t packed = worker.countTensor.GetValue(localToken);
        int32_t survivorCount = packed & 0xFF;
        int32_t fullCount = (packed >> 8) & 0xFF;
        int32_t fullMinK = (packed >> 16) & 0xFF;
        ++fullCount;
        fullMinK = k < fullMinK ? k : fullMinK;
        // 行号越过 maxOutputSize 的成员被 clamp 裁掉：只计入全量口径，不进成员表。
        if (row < static_cast<int32_t>(worker.maxOutputSize)) {
            const int32_t tileIdx = (row - expertRowBegin) / static_cast<int32_t>(L1_TILE_M_256);
            const int32_t flagSlotOffset =
                static_cast<int32_t>(expertIdx) * worker.dispatchFlagSlotCountPerExpert + tileIdx * INT_CACHELINE;
            const uint32_t memberBase =
                static_cast<uint32_t>(localToken) * worker.topK * 3U + static_cast<uint32_t>(survivorCount) * 3U;
            worker.memberTensor.SetValue(memberBase, row);
            worker.memberTensor.SetValue(memberBase + 1U, topkIndex);
            worker.memberTensor.SetValue(memberBase + 2U, flagSlotOffset);
            ++survivorCount;
        }
        worker.countTensor.SetValue(localToken, survivorCount | (fullCount << 8) | (fullMinK << 16));
    }
}

// 扫描在途槽并释放其半区（等装载完成 → 扫描 → 武装"可复写"）。无在途时为空操作。
template <typename TopkIndexType>
__aicore__ inline void DedupFlushPendingSlot(DedupPrepassWorker<TopkIndexType> &worker, DedupPendingSlot &pending,
                                             const DedupChunkWindow &window)
{
    if (pending.expertIdx < 0) {
        return;
    }
    const TEventID evt = DedupRouteHalfEvent(pending.half);
    WaitFlag<AscendC::HardEvent::MTE2_S>(evt);
    LocalTensor<TopkIndexType> seg =
        worker.routeLoadTensor[pending.half * static_cast<uint32_t>(worker.ROUTE_HALF_ITEMS)];
    DedupPrepassScanSlot(worker, seg, window, pending.slotCount, pending.rowBase, 0,
                         static_cast<uint32_t>(pending.expertIdx), pending.expertRowBegin);
    SetFlag<AscendC::HardEvent::S_MTE2>(evt);
    pending.expertIdx = -1;
}

// 大槽（单槽超过半区容量，大 bs 罕见路径）：原地分段串行扫描，复用当前装载半区。
// 段末值低于窗口下界可整段跳过，达到上界即提前结束（槽内升序）。
template <typename TopkIndexType>
__aicore__ inline void DedupScanLargeSlot(DedupPrepassWorker<TopkIndexType> &worker, __gm__ TopkIndexType *slotPtr,
                                          int32_t slotCount, uint32_t loadHalf, const DedupChunkWindow &window,
                                          int32_t rowBase, uint32_t expertIdx, int32_t expertRowBegin)
{
    const TEventID evt = DedupRouteHalfEvent(loadHalf);
    LocalTensor<TopkIndexType> seg = worker.routeLoadTensor[loadHalf * static_cast<uint32_t>(worker.ROUTE_HALF_ITEMS)];
    bool reachedEnd = false;
    for (int32_t segBase = 0; segBase < slotCount && !reachedEnd; segBase += worker.ROUTE_HALF_ITEMS) {
        const int32_t segCount =
            (slotCount - segBase) < worker.ROUTE_HALF_ITEMS ? (slotCount - segBase) : worker.ROUTE_HALF_ITEMS;
        GlobalTensor<TopkIndexType> slotGm;
        slotGm.SetGlobalBuffer(slotPtr + segBase);
        WaitFlag<AscendC::HardEvent::S_MTE2>(evt); // 本半区上次扫描已完成，可复写
        DataCopyExtParams segCopyParams{1U, static_cast<uint32_t>(segCount * sizeof(TopkIndexType)), 0U, 0U, 0U};
        DataCopyPadExtParams<TopkIndexType> segCopyPad{false, 0U, 0U, 0U};
        DataCopyPad(seg, slotGm, segCopyParams, segCopyPad);
        SetFlag<AscendC::HardEvent::MTE2_S>(evt);
        WaitFlag<AscendC::HardEvent::MTE2_S>(evt);
        if (seg.GetValue(static_cast<uint32_t>(segCount - 1)) >= window.targetLo) {
            const int32_t lastVal = seg.GetValue(static_cast<uint32_t>(segCount - 1));
            DedupPrepassScanSlot(worker, seg, window, segCount, rowBase, segBase, expertIdx, expertRowBegin);
            if (lastVal >= window.targetHi) {
                reachedEnd = true;
            }
        }
        SetFlag<AscendC::HardEvent::S_MTE2>(evt);
    }
}

/*
 * 扫描一个工作项覆盖的全部 route 槽（[expert][srcRank]）。
 * 槽间 MTE2 双缓冲：单段槽先发起装载、再扫描上一在途槽，装载与扫描重叠；大槽走独立串行路径。
 * 事件生命周期在本函数内闭环：入口武装两半 S_MTE2，出口消费（每半区两方向：
 * S_MTE2=扫描完可复写、MTE2_S=装载完可扫描），零悬挂。
 */
template <typename TopkIndexType, typename DispatchScratch>
__aicore__ inline void DedupScanRouteSlots(DedupPrepassWorker<TopkIndexType> &worker,
                                           const MoeStageCommonConfig &common, const Params &params,
                                           DispatchScratch &scratch, uint32_t srcRank, const DedupChunkWindow &window)
{
    const uint64_t routeIndexAlignSize = static_cast<uint64_t>(CalcDispatchRouteIndexAlignSize(params.tilingData));
    SetFlag<AscendC::HardEvent::S_MTE2>(static_cast<TEventID>(SYNC_EVENT_ID4));
    SetFlag<AscendC::HardEvent::S_MTE2>(static_cast<TEventID>(SYNC_EVENT_ID6));
    uint32_t loadHalf = 0U;
    DedupPendingSlot pending;
    for (uint32_t expertIdx = 0U; expertIdx < common.moeExpertPerRank; ++expertIdx) {
        const uint32_t prefixIdx = expertIdx * common.worldSize + srcRank;
        const int32_t rowBase = prefixIdx == 0U ? 0 : scratch.cumsumInfoTensor.GetValue(prefixIdx - 1U);
        const int32_t slotCount = scratch.cumsumInfoTensor.GetValue(prefixIdx) - rowBase;
        if (slotCount <= 0) {
            continue;
        }
        const int32_t expertRowBegin =
            expertIdx == 0U ? 0 : scratch.cumsumInfoTensor.GetValue(expertIdx * common.worldSize - 1U);
        __gm__ TopkIndexType *slotPtr = reinterpret_cast<__gm__ TopkIndexType *>(
            params.peermemInfo.maskRecvPtr + static_cast<uint64_t>(prefixIdx) * routeIndexAlignSize);
        if (slotCount > worker.ROUTE_HALF_ITEMS) {
            DedupFlushPendingSlot(worker, pending, window); // 大槽独占半区前先清空在途
            DedupScanLargeSlot(worker, slotPtr, slotCount, loadHalf, window, rowBase, expertIdx, expertRowBegin);
            continue;
        }
        // 先发起本槽装载，再扫描在途槽：装载的 MTE2 与扫描的标量操作重叠。
        const TEventID loadEvt = DedupRouteHalfEvent(loadHalf);
        LocalTensor<TopkIndexType> loadSeg =
            worker.routeLoadTensor[loadHalf * static_cast<uint32_t>(worker.ROUTE_HALF_ITEMS)];
        GlobalTensor<TopkIndexType> slotGm;
        slotGm.SetGlobalBuffer(slotPtr);
        WaitFlag<AscendC::HardEvent::S_MTE2>(loadEvt);
        DataCopyExtParams slotCopyParams{1U, static_cast<uint32_t>(slotCount * sizeof(TopkIndexType)), 0U, 0U, 0U};
        DataCopyPadExtParams<TopkIndexType> slotCopyPad{false, 0U, 0U, 0U};
        DataCopyPad(loadSeg, slotGm, slotCopyParams, slotCopyPad);
        SetFlag<AscendC::HardEvent::MTE2_S>(loadEvt);
        DedupFlushPendingSlot(worker, pending, window);
        pending.expertIdx = static_cast<int32_t>(expertIdx);
        pending.rowBase = rowBase;
        pending.slotCount = slotCount;
        pending.expertRowBegin = expertRowBegin;
        pending.half = loadHalf;
        loadHalf ^= 1U;
    }
    DedupFlushPendingSlot(worker, pending, window);
    WaitFlag<AscendC::HardEvent::S_MTE2>(static_cast<TEventID>(SYNC_EVENT_ID4));
    WaitFlag<AscendC::HardEvent::S_MTE2>(static_cast<TEventID>(SYNC_EVENT_ID6));
}

// 落一条 PLAIN 行 desc（该 token 全量只出现一次，dispatch/combine 走原路径）：word0 = 0。
template <typename TopkIndexType>
__aicore__ inline void DedupStorePlainRowDesc(DedupPrepassWorker<TopkIndexType> &worker, uint32_t memberBase)
{
    const int32_t row = worker.memberTensor.GetValue(memberBase);
    const TEventID evt = DedupDescHalfEvent(worker.descHalf);
    LocalTensor<int32_t> asmHalf = worker.descAsmTensor[worker.descHalf * worker.DESC_ASM_HALF_INT32];
    WaitFlag<AscendC::HardEvent::MTE3_S>(evt); // 该半区上次 GM 写已完成，可复写
    asmHalf.SetValue(0U, 0);
    SetFlag<AscendC::HardEvent::S_MTE3>(evt);
    WaitFlag<AscendC::HardEvent::S_MTE3>(evt);
    DataCopy(worker.rowDescGm[static_cast<uint64_t>(row) * worker.strideInt32], asmHalf, INT32_PER_256B);
    SetFlag<AscendC::HardEvent::MTE3_S>(evt); // 写完武装，下次复用本半区时消费
    worker.descHalf ^= 1U;
}

// 落一个重复组的全部成员行 desc。组成员按扫描序 = 专家升序 = 行号升序，天然满足 FIRST/LAST 语义；
// 组内每个成员行持有整组完整拷贝，读一行即可拿到全组信息。
template <typename TopkIndexType>
__aicore__ inline void DedupStoreGroupRowDesc(DedupPrepassWorker<TopkIndexType> &worker, uint32_t memberBase,
                                              int32_t survivorCount, int32_t fullMinK)
{
    for (int32_t m = 0; m < survivorCount; ++m) {
        int32_t flags = DEDUP_ROW_FLAG_GROUP | survivorCount;
        flags |= (m == 0 ? DEDUP_ROW_FLAG_FIRST : 0);
        flags |= (m == survivorCount - 1 ? DEDUP_ROW_FLAG_LAST : 0);
        const TEventID evt = DedupDescHalfEvent(worker.descHalf);
        LocalTensor<int32_t> asmHalf = worker.descAsmTensor[worker.descHalf * worker.DESC_ASM_HALF_INT32];
        WaitFlag<AscendC::HardEvent::MTE3_S>(evt); // 该半区上次 GM 写已完成，可复写
        asmHalf.SetValue(0U, flags);
        // 两个半区严格交替，本 token 内每个半区只在首次使用（m<2）时拼整表，之后复用只改 word0，
        // 标量拼装从每 token s 次降到至多 2 次。不得取消该复用：desc 落盘曾是 prepass 的第一大头。
        if (m < 2) {
            asmHalf.SetValue(1U, fullMinK);
            for (int32_t survivorIdx = 0; survivorIdx < survivorCount; ++survivorIdx) {
                const uint32_t srcBase = memberBase + static_cast<uint32_t>(survivorIdx) * 3U;
                const uint32_t dstBase = DEDUP_ROW_DESC_HEADER_INT32 + static_cast<uint32_t>(survivorIdx) * 3U;
                asmHalf.SetValue(dstBase, worker.memberTensor.GetValue(srcBase));
                asmHalf.SetValue(dstBase + 1U, worker.memberTensor.GetValue(srcBase + 1U));
                asmHalf.SetValue(dstBase + 2U, worker.memberTensor.GetValue(srcBase + 2U));
            }
        }
        const int32_t row = worker.memberTensor.GetValue(memberBase + static_cast<uint32_t>(m) * 3U);
        SetFlag<AscendC::HardEvent::S_MTE3>(evt);
        WaitFlag<AscendC::HardEvent::S_MTE3>(evt);
        DataCopy(worker.rowDescGm[static_cast<uint64_t>(row) * worker.strideInt32], asmHalf, worker.strideInt32);
        SetFlag<AscendC::HardEvent::MTE3_S>(evt);
        worker.descHalf ^= 1U;
    }
}

// 按扫描结果逐 token 落 rowDesc。fullCount>=2 即使幸存成员只剩 1 个也按组发
// （minK 取全量口径，保证发送槽与归属卡 canonical 槽一致）；fullCount==1 才走 PLAIN。
template <typename TopkIndexType>
__aicore__ inline void DedupStoreRowDescChunk(DedupPrepassWorker<TopkIndexType> &worker, int32_t chunkTokenCount)
{
    for (int32_t localToken = 0; localToken < chunkTokenCount; ++localToken) {
        const int32_t packed = worker.countTensor.GetValue(localToken);
        const int32_t survivorCount = packed & 0xFF;
        const int32_t fullCount = (packed >> 8) & 0xFF;
        const int32_t fullMinK = (packed >> 16) & 0xFF;
        if (survivorCount <= 0) {
            continue;
        }
        const uint32_t memberBase = static_cast<uint32_t>(localToken) * worker.topK * 3U;
        if (fullCount == 1) {
            DedupStorePlainRowDesc(worker, memberBase);
        } else {
            DedupStoreGroupRowDesc(worker, memberBase, survivorCount, fullMinK);
        }
    }
}

// 建表工人集按档切换（判据见 DEDUP_AIV0_PREPASS_MAX_HIDDEN_DIM）：AIV0 参与档按
// AIV1=jobIndex / AIV0=jobIndex+totalJobs 平铺；不参与档保持 AIV1 单方建表。
// 返回 false = 本子核不参与建表。
__aicore__ inline bool GetDedupPrepassWorkerSlot(const BlockJobContext &blockJob, const Params &params,
                                                 uint32_t &workerIndex, uint32_t &totalWorkers)
{
    const bool aiv0Joins = params.tilingData->hiddenDim <= DEDUP_AIV0_PREPASS_MAX_HIDDEN_DIM;
    if (GetSubBlockIdx() != 1U && !aiv0Joins) {
        return false;
    }
    workerIndex = GetSubBlockIdx() == 1U ? blockJob.jobIndex : blockJob.jobIndex + blockJob.totalJobs;
    totalWorkers = aiv0Joins ? blockJob.totalJobs * 2U : blockJob.totalJobs;
    return true;
}

// 计算工作项 chunkIdx 的 token 窗口（末段按 numTokens 截断）。
__aicore__ inline DedupChunkWindow MakeDedupChunkWindow(uint32_t chunkIdx, uint32_t chunkTokens, uint32_t numTokens,
                                                        uint32_t topK, int32_t &tokenHi)
{
    DedupChunkWindow window;
    window.tokenLo = static_cast<int32_t>(chunkIdx * chunkTokens);
    tokenHi = window.tokenLo + static_cast<int32_t>(chunkTokens);
    if (tokenHi > static_cast<int32_t>(numTokens)) {
        tokenHi = static_cast<int32_t>(numTokens);
    }
    window.targetLo = window.tokenLo * static_cast<int32_t>(topK);
    window.targetHi = tokenHi * static_cast<int32_t>(topK);
    return window;
}

// 执行一个 prepass 工作项：count 清零 → 槽扫描 → desc 落盘 → drain desc 乒乓并置本项 ready。
template <typename TopkIndexType, typename DispatchScratch>
__aicore__ inline void RunDedupPrepassWorkItem(DedupPrepassWorker<TopkIndexType> &worker,
                                               const MoeStageCommonConfig &common, const Params &params,
                                               DispatchScratch &scratch, const DedupConfig &dedup, uint32_t workItem)
{
    const uint32_t srcRank = workItem / dedup.prepassChunkCount;
    const uint32_t chunkIdx = workItem % dedup.prepassChunkCount;
    int32_t tokenHi = 0;
    const DedupChunkWindow window = MakeDedupChunkWindow(chunkIdx, dedup.prepassChunkTokens,
                                                         params.tilingData->numMaxTokensPerRank, worker.topK, tokenHi);
    // 上一工作项的标量 count 写先于本项向量清零（S→V），再保证清零结果对标量可见（V→S）。
    SyncFuncStatic<AscendC::HardEvent::S_V, SYNC_EVENT_ID2>();
    Duplicate<int32_t>(worker.countTensor, DEDUP_PACK_INIT, static_cast<int32_t>(dedup.prepassChunkTokens));
    SyncFuncStatic<AscendC::HardEvent::V_S, SYNC_EVENT_ID2>();
    DedupScanRouteSlots(worker, common, params, scratch, srcRank, window);
    DedupStoreRowDescChunk(worker, tokenHi - window.tokenLo);
    // ready 置位前 drain desc 乒乓两半（末尾 1-2 行的 GM 写可能还在飞，而 ready 语义 = 本项
    // desc 全部落盘），随即重新武装供下一工作项。
    WaitFlag<AscendC::HardEvent::MTE3_S>(static_cast<TEventID>(SYNC_EVENT_ID3));
    WaitFlag<AscendC::HardEvent::MTE3_S>(static_cast<TEventID>(SYNC_EVENT_ID5));
    SetFlag<AscendC::HardEvent::MTE3_S>(static_cast<TEventID>(SYNC_EVENT_ID3));
    SetFlag<AscendC::HardEvent::MTE3_S>(static_cast<TEventID>(SYNC_EVENT_ID5));
    __gm__ int32_t *readyBase = reinterpret_cast<__gm__ int32_t *>(dedup.prepassReadyPtr);
    AtomicAdd(readyBase + (1U + workItem) * INT_CACHELINE, static_cast<int32_t>(1));
}

/*
 * 去重 prepass 主入口：扫描 route 槽，为每个接收行写一条 rowDesc（布局见 mega_moe_constants.h）。
 * 分工：工作项 = (srcRank, token 段)，全体 AIV1+AIV0 平铺认领（AIV0 的本地前缀表与 UB 布局由
 * PrepareDedupPrepassAiv0Local 先行就位）。同一 (srcRank, token) 的所有出现恒落在同一工作项内，
 * 工作项之间零交叠，无需原子。
 * 依赖：count 表已就绪（cumsumInfoTensor；count 到达即 route 槽到达，是 dispatch 沿用的既有契约）。
 * 不得把 prepass 整体挪给 AIV0 提前跑：大 bs 下 AIV0 早期密集访问 route 窗口会触发 aicore trap
 * （驱动侧机制未明，实测复现），当前平铺形态是稳定边界。
 */
template <typename TopkIndexType, typename DispatchScratch>
__aicore__ inline void BuildDedupRowDescTable(const DedupConfig &dedup, const MoeStageCommonConfig &common,
                                              const BlockJobContext &blockJob, const MoeSyncWorkspaceLayout &syncLayout,
                                              const Params &params, DispatchScratch &scratch)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    if (blockJob.totalJobs == 0U) {
        return;
    }
    uint32_t workerIndex = 0U;
    uint32_t totalWorkers = 0U;
    if (!GetDedupPrepassWorkerSlot(blockJob, params, workerIndex, totalWorkers)) {
        return;
    }

    // 切片宽/项数以 DedupConfig 缓存值为准（与 dispatch 消费侧同一 CalcDedupPrepassChunkTokens 单源）。
    const uint32_t totalWorkItems = common.worldSize * dedup.prepassChunkCount;
    __gm__ int32_t *readyBase = reinterpret_cast<__gm__ int32_t *>(dedup.prepassReadyPtr);

    DedupPrepassWorker<TopkIndexType> worker =
        CreateDedupPrepassWorker<TopkIndexType>(dedup, syncLayout, params, scratch);
    // 武装 desc 乒乓两半的 MTE3_S；每个工作项尾 drain 后重新武装，函数尾统一消费，零悬挂。
    SetFlag<AscendC::HardEvent::MTE3_S>(static_cast<TEventID>(SYNC_EVENT_ID3));
    SetFlag<AscendC::HardEvent::MTE3_S>(static_cast<TEventID>(SYNC_EVENT_ID5));

    for (uint32_t workItem = workerIndex; workItem < totalWorkItems; workItem += totalWorkers) {
        RunDedupPrepassWorkItem(worker, common, params, scratch, dedup, workItem);
    }
    // 消费掉最后一次武装，不留悬挂 flag 毒化后续同 id 同步。
    WaitFlag<AscendC::HardEvent::MTE3_S>(static_cast<TEventID>(SYNC_EVENT_ID3));
    WaitFlag<AscendC::HardEvent::MTE3_S>(static_cast<TEventID>(SYNC_EVENT_ID5));

    // 槽 0 = done 计数，AIV1 计低 16 位、AIV0 计高 16 位。dispatch 发布 gate 只等高位（全体 AIV0）：
    // 每个 AIV1 发布前自己必然已建完，AIV1 互等会打断 prepass 与 dispatch 的流水交叠（实测大形状
    // 明显劣化）；combine 兜底等全值。本核不等任何人。
    AtomicAdd(readyBase, GetSubBlockIdx() == 1U ? static_cast<int32_t>(1) : static_cast<int32_t>(1) << 16);
}

} // namespace MegaMoeImpl

#endif // MEGA_MOE_TOKEN_DEDUP_H
