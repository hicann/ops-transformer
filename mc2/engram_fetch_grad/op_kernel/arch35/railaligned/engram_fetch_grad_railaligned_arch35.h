/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file engram_fetch_grad_railaligned_arch35.h
 * \brief EngramFetchGrad 分轨（rail-aligned）kernel 主类。
 *        三阶段：A 全核（barrier 两轮清区 + unsort + 桶表 + 前缀）→ B 三组并发
 *        （S 统一发送 / F 消费+暂存 / O 收集）→ C 全核（排序 + 去重累加）。
 *        S 核统一发送（通道单生产者：hcomm PostSend 的 sqHead 非原子、Mutex 仅核内，
 *        多核并发会覆盖 SQ 槽）——目的 → 属主 S 核固定映射（d mod sendCoreCnt），
 *        每对端全部远程写同一核同一条流；F 纯消费 + 暂存环；O 收集。
 *        感知只取一数（ranksPerSupernode，attr 透传），超节点数全推导。
 *        不 include 老通信头；排序复用 SortLib、去重复用 EngramFetchGradUnique。
 */

#ifndef ENGRAM_FETCH_GRAD_RAILALIGNED_ARCH35_H
#define ENGRAM_FETCH_GRAD_RAILALIGNED_ARCH35_H

#include "../../engram_fetch_grad_tiling_data.h"
#include "../../engram_fetch_grad_utils.h"
#include "engram_fetch_grad_route.h"
#include "engram_fetch_grad_win.h"
#include "../sort/sort_lib.h"
#include "../engram_fetch_grad_unique.h"
#include "adv_api/hccl/hccl.h"

namespace Mc2Kernel {

class EngramFetchGradRailAlignedArch35 {
public:
    __aicore__ inline EngramFetchGradRailAlignedArch35() = default;

    __aicore__ inline void Init(GM_ADDR commContext, GM_ADDR gradFetched, GM_ADDR perm, GM_ADDR sendCounts,
                                GM_ADDR recvCounts, GM_ADDR recvLocalEntry, GM_ADDR numRecv, GM_ADDR gradUniqueOut,
                                GM_ADDR uniqueLocalEntryOut, GM_ADDR numUniqueOut, GM_ADDR workspaceGM, TPipe* pipe,
                                const EngramFetchGradTilingData* tilingData);

    __aicore__ inline void Process();

private:
    template <AscendC::HardEvent event>
    __aicore__ inline void SyncEv()
    {
        int32_t eventID = static_cast<int32_t>(tpipe_->FetchEventID(event));
        AscendC::SetFlag<event>(eventID);
        AscendC::WaitFlag<event>(eventID);
    }

    __aicore__ inline int32_t Rd32(GM_ADDR p)
    {
        return AscendC::ReadGmBypassDCache(reinterpret_cast<__gm__ int32_t*>(p));
    }

    __aicore__ inline uint64_t Rd64(GM_ADDR p) // 两个 int32 bypass 读拼 u64（槽头 handle/area）
    {
        uint32_t lo = static_cast<uint32_t>(Rd32(p));
        uint32_t hi = static_cast<uint32_t>(Rd32(p + 4));
        return (static_cast<uint64_t>(hi) << 32) | lo;
    }

    __aicore__ inline void Wr64(GM_ADDR p, uint64_t v) // 槽头写（初始化一次，无热路径）
    {
        Wr32(p, static_cast<int32_t>(v & 0xFFFFFFFFULL));
        Wr32(p + 4, static_cast<int32_t>(v >> 32));
    }

    __aicore__ inline void Wr32(GM_ADDR p, int32_t v) // 标量 GM 写：UB 暂存 + 4B DataCopy
    {
        LocalTensor<int32_t> t = creditBuf_.Get<int32_t>();
        t.SetValue(0, v);
        SyncEv<AscendC::HardEvent::S_MTE3>();
        GlobalTensor<int32_t> gm;
        gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(p));
        AscendC::DataCopyParams params = {1U, 4U, 0U, 0U};
        AscendC::DataCopyPad(gm, t, params);
        SyncEv<AscendC::HardEvent::MTE3_S>();
    }

    __aicore__ inline void LoadTilingData(const EngramFetchGradTilingData* t);
    __aicore__ inline void CalcWorkSpaceOffset(const EngramFetchGradTilingData* t);
    __aicore__ inline void InitBuffers(const EngramFetchGradTilingData* t);
    __aicore__ inline void InitCoreGroups();
    __aicore__ inline uint32_t ReadNumRecv();

    // ── barrier 两轮协议（W 槽 + epoch 槽）──
    __aicore__ inline uint32_t PeerOfWriterSlot(uint32_t slot); // 槽 → 写者全局 rank（自槽返回 0xFFFFFFFF）
    __aicore__ inline bool IsSelfWriterSlot(uint32_t slot);     // 同伴 myL / 轨道 mySn 两自槽
    __aicore__ inline void BarrierWaitWritersDone(uint32_t epoch);  // 轮①：等 W 槽 ≥ 2c
    __aicore__ inline void BarrierAnnounceCleared(uint32_t epoch);  // 轮②：向写者通告 2c+1
    __aicore__ inline void BarrierWaitPeersCleared(uint32_t epoch); // 发送门：等 W 槽 ≥ 2c+1
    __aicore__ inline void BarrierWriteCallDone(uint32_t epoch);    // 调用末：向写者写 2(c+1)，核0 推 epoch

    // ── 阶段 A ──
    __aicore__ inline void ClearRecvAreas();
    __aicore__ inline void ClearGmRange(GM_ADDR base, int64_t r0, int64_t r1);
    __aicore__ inline void UnsortGradRail();
    __aicore__ inline uint32_t LoadGradChunk(int64_t pos, int64_t end, LocalTensor<uint8_t>& buf, int32_t bufIdx,
                                             LocalTensor<int32_t>& idxUb, uint32_t tokensPerBuf);
    __aicore__ inline void StoreGradChunk(int64_t base, uint32_t count, LocalTensor<uint8_t>& buf, int32_t bufIdx);
    __aicore__ inline void BuildSendPrefix(); // 全核并行两遍（segCount/coreStart 作 scratch）
    __aicore__ inline void BuildRecvPrefix();
    __aicore__ inline void MarkOwnerSorted();       // 二分 sendPrefix + 区间填充
    __aicore__ inline void BucketClassify();        // 全核条带段表（dst 升序 → 行天然连续段）
    __aicore__ inline void MergeSegmentsBySGroup(); // S0 合并同 dst 段 → 任务表（write/done 计数唯一）

    // ── 阶段 B：三组（S/F/O）──
    __aicore__ inline void SendLoopS();
    __aicore__ inline void FwdLoopF();
    __aicore__ inline void OwnerLoopO();
    __aicore__ inline void CollectOneRow(GM_ADDR rowDataGM, int32_t srcRank, uint32_t& cursor);

    // ── 阶段 C ──
    __aicore__ inline void SoftBarrier();
    __aicore__ inline void RunSortPhase(uint32_t collected);

    // ── 成员 ──
    __gm__ EngramRailCtxHead* ctxHead_{nullptr}; // 平铺 rail ctx 头（node 表经 NodeWinAt/NodeChannelAt 访问）
    RankCoord me_{};
    RailTopo topo_{}; // ranksPerSupernode 感知（attr）+ 超节点数/轨道数推导（kernel 唯一权威）
    uint32_t myGlobalRank_{0};
    WinLayout win_{};
    GM_ADDR localWin_{nullptr}; // = NodeWinAt(ctx, localId)（本卡 WIN 基址；node 表自槽）
    uint32_t aivId_{0};
    uint32_t totalBlocks_{1};
    int32_t numEntriesPerRank_{0};
    int64_t numTokens_{0};
    int64_t hiddenBytes_{0};
    int64_t hiddenDim_{0};
    int64_t cap_{0}; // O/F 收集数组容量 = recv_local_entry.dim0 = tilingData.totalRecv
    uint32_t batchRows_{1};
    int32_t inputDtype_{0};
    int32_t outputDtype_{0};
    uint32_t tileBytes_{TILE_BYTES};
    uint32_t gradSubBatch_{GRAD_SUB_BATCH};
    uint32_t chunkElems_{0};
    uint32_t uniqueEntryBytes_{0};
    uint32_t sortNumTileData_{0};
    uint32_t sortTmpUbSize_{0};

    // 输入/输出（GM_ADDR 展开含 *，只能单声明符一行——多声明符会把后续字段退化成 __gm__ uint8_t）
    GM_ADDR gradFetchedGM_{nullptr};
    GM_ADDR permGM_{nullptr};
    GM_ADDR sendCountsGM_{nullptr};
    GM_ADDR recvCountsGM_{nullptr};
    GM_ADDR recvLocalEntryGM_{nullptr};
    GM_ADDR numRecvGM_{nullptr};
    GM_ADDR gradUniqueOutGM_{nullptr};
    GM_ADDR uniqueLocalEntryOutGM_{nullptr};
    GM_ADDR numUniqueOutGM_{nullptr};
    GM_ADDR workspaceGM_{nullptr};

    // workspace 段（与 engram_fetch_grad_tiling.cpp 分轨段 SetRailWorkSpace 逐段一致）
    GM_ADDR gradSortedGM_{nullptr};
    GM_ADDR ownerSortedGM_{nullptr};
    GM_ADDR bucketSegGM_{nullptr};
    GM_ADDR relayGM_{nullptr}; // F 核暂存环：[fCoreCnt][ranksPerSupernode+S 槽][batchRows×P]——F 私有格
    GM_ADDR relayStagedGM_{nullptr}; // [fCoreCnt][ranksPerSupernode+S]：F 已备行数（滚动累计，F 单写）
    GM_ADDR relayDrainedGM_{nullptr}; // [fCoreCnt][ranksPerSupernode+S]：属主 S 核已发行数（滚动游标，S 单写）
    GM_ADDR relayEofGM_{nullptr}; // [fCoreCnt][ranksPerSupernode+S]：EOF 标记（F 相位完结后写 1）
    GM_ADDR sStageGM_{nullptr};
    GM_ADDR collectEntryGM_{nullptr};
    GM_ADDR collectRowsGM_{nullptr};
    GM_ADDR counterScratchGM_{nullptr};
    GM_ADDR srcArriveGM_{nullptr};
    GM_ADDR recvPrefixGM_{nullptr};
    GM_ADDR sendPrefixGM_{nullptr};
    GM_ADDR flagScratchGM_{nullptr};
    GM_ADDR segCountGM_{nullptr};
    GM_ADDR coreStartGM_{nullptr};
    GM_ADDR sortCompanionGM_{nullptr};
    GM_ADDR sortWorkspaceGm_{nullptr};
    GM_ADDR sortBarrierGM_{nullptr};
    // bucketSeg 内部布局
    GM_ADDR segCountsGM_{nullptr}; // 每核段数头（含 taskCount 在最前）
    GM_ADDR stripsGM_{nullptr};    // 核条带
    GM_ADDR taskTableGM_{nullptr}; // 任务表 (kind,dst,start,cnt)×4 int32
    uint32_t stripCap_{0};

    // 三组边界
    uint32_t sendCoreEnd_{0};
    uint32_t fwdCoreEnd_{0};

    // UB
    AscendC::TPipe* tpipe_{nullptr};
    TBuf<> hcommBuf_;
    TBuf<> creditBuf_;
    TBuf<> areaHeadBuf_;
    TBuf<> rowBlockBuf_;
    TBuf<> indicesBuf_;
    TBuf<> statusBuf_;
    TBuf<> tempBuf_;
    TBuf<> entryBuf_;
    TBuf<> gradBuf_;
    TBuf<> offBuf_;
    TBuf<> flowBuf_;
    TBuf<> sFlowBuf_;
    TBuf<> castFp32Buf_;
    TBuf<> accumBuf_;
    uint32_t indicesBufElems_{0};
    uint32_t flowBufBytes_{0};
    uint32_t statusBufBytes_{0};
    uint32_t tempBufBytes_{0};
    uint32_t indicesBufBytes_{0};
    uint32_t castBufBytes_{0};
    uint32_t accumBufBytes_{0};
    AscendC::Hcomm<COMM_PROTOCOL_UBC_CTP> hcomm_;
    EngramFetchGradUnique::EngramFetchGradUnique uniqueScatter_;
    int32_t evMte2ToMte3_[2] = {0, 0}; // unsort 乒乓
    int32_t evMte3ToMte2_[2] = {0, 0};
    int32_t evCopy2To3_[2] = {0, 0}; // 阶段 A 行搬运乒乓
    int32_t evCopy3To2_[2] = {0, 0};
    uint32_t softBarrierGen_{0};
    uint32_t softBarrierPageInts_{0};
    static constexpr uint32_t SOFT_BARRIER_MAX_GENS = 16U;
    static constexpr uint32_t SPIN_PAUSE_ITERS = 64U;
    // 超时统一走 win.h 的真墙钟 WIN_TIMEOUT_US（60s）

    friend struct EngramRailSortSoftBarrier;
};

struct EngramRailSortSoftBarrier {
    EngramFetchGradRailAlignedArch35* self;
    __aicore__ inline void operator()() const
    {
        self->SoftBarrier();
    }
};

// ═══════════════ 初始化 ═══════════════

__aicore__ inline uint32_t EngramFetchGradRailAlignedArch35::ReadNumRecv()
{
    GlobalTensor<int32_t> numRecvGM;
    numRecvGM.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(numRecvGM_));
    AscendC::DataCacheCleanAndInvalid<int32_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(
        numRecvGM);
    return static_cast<uint32_t>(numRecvGM.GetValue(0));
}

/// @brief tilingData 装载与运行期派生：拓扑、WIN 布局、批行数钳制
__aicore__ inline void EngramFetchGradRailAlignedArch35::LoadTilingData(const EngramFetchGradTilingData* t)
{
    numEntriesPerRank_ = t->numEntriesPerRank;
    numTokens_ = t->numTokens;
    hiddenBytes_ = t->hiddenBytes;
    hiddenDim_ = t->hiddenDim;
    inputDtype_ = t->inputDtype;
    outputDtype_ = t->outputDtype;
    batchRows_ = t->batchRows;
    cap_ = t->totalRecv; // 容量 = recv_local_entry.dim0（前向记账直接提供，零新增 attr）
    sortNumTileData_ = t->sortNumTileData;
    sortTmpUbSize_ = t->sortTmpUbSize;
    chunkElems_ = t->chunkElems;
    gradSubBatch_ = t->gradSubBatch;
    topo_ = MakeRailTopo(t->ranksPerSupernode, t->rankSize);
    win_ = DeriveWinLayout(topo_, hiddenBytes_, t->numRows);
    // 批行数防御钳制：blockLen 为 uint16，满批 rows×P 须 ≤ MAX_BLOCK_BYTES；对 tiling
    // 值再钳制（错配兜底），保 >=1 下限防除零与零行批忙转
    {
        if (win_.rowStride > 0U) {
            const uint32_t maxRows = MAX_BLOCK_BYTES / win_.rowStride;
            if (batchRows_ > maxRows) {
                batchRows_ = maxRows;
            }
        }
        if (batchRows_ == 0U) {
            batchRows_ = 1U;
        }
        ascendc_assert(batchRows_ > 0U, "rail grad batchRows must be positive");
    }
    if (chunkElems_ > 0U) {
        uint32_t chunkFp32Stride = (chunkElems_ * sizeof(float) + UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN;
        uniqueEntryBytes_ = FLUSH_CAST_HEAD_BYTES + ((outputDtype_ != ENGRAM_DT_FLOAT) ? 2U * chunkFp32Stride : 0U);
    }
}

/// @brief workspace 各段基址计算（与 SetRailWorkSpace 逐段一致）
__aicore__ inline void EngramFetchGradRailAlignedArch35::CalcWorkSpaceOffset(const EngramFetchGradTilingData* t)
{
    const int64_t T = numTokens_;
    const int64_t HB = hiddenBytes_;
    const uint32_t aiv = static_cast<uint32_t>(t->aivNum);
    const int64_t world = static_cast<int64_t>(t->rankSize);
    const int64_t cap = cap_;
    const uint32_t ranksPerSupernode = topo_.ranksPerSupernode;
    const uint32_t P = win_.rowStride;
    auto align32 = [](int64_t x) __attribute__((cce_aicore))
    {
        return (x + UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN;
    };

    int64_t off = 0;
    gradSortedGM_ = workspaceGM_ + off;
    off += T * HB;
    ownerSortedGM_ = workspaceGM_ + off;
    off += align32(T * 4);
    int64_t stripCap = (T + aiv - 1) / aiv + 2; // 每核段数上界：行/核 + 边界分裂
    stripCap_ = static_cast<uint32_t>(stripCap);
    int64_t bucketSegBytes = align32((1 + aiv) * 4) + static_cast<int64_t>(aiv) * stripCap * 12 + (T + aiv) * 16;
    bucketSegGM_ = workspaceGM_ + off;
    off += align32(bucketSegBytes);
    segCountsGM_ = bucketSegGM_; // [0]=taskCount, [1+c]=核 c 段数
    stripsGM_ = bucketSegGM_ + align32((1 + aiv) * 4);
    taskTableGM_ = stripsGM_ + static_cast<int64_t>(aiv) * stripCap * 12;
    // ═══ S 核统一发送：F 暂存环 + 三张控制表 ═══
    // 暂存环 [fCoreCnt][slots=ranksPerSupernode][batchRows×P]：每 (F核, owner 槽) 一个 SPSC 环格
    const uint32_t relaySlots = ranksPerSupernode;
    const uint32_t fCoreCnt = (t->fwdCoreCnt > 0U) ? t->fwdCoreCnt : 1U;
    relayGM_ = workspaceGM_ + off;
    off += static_cast<int64_t>(fCoreCnt) * relaySlots * batchRows_ * P;
    const int64_t relayCtlInts = static_cast<int64_t>(fCoreCnt) * relaySlots;
    relayStagedGM_ = workspaceGM_ + off;
    off += align32(relayCtlInts * 4);
    relayDrainedGM_ = workspaceGM_ + off;
    off += align32(relayCtlInts * 4);
    relayEofGM_ = workspaceGM_ + off;
    off += align32(relayCtlInts * 4);
    sStageGM_ = workspaceGM_ + off;
    off += static_cast<int64_t>(t->sendCoreCnt) * batchRows_ * P;
    collectEntryGM_ = workspaceGM_ + off;
    off += align32(cap * 4);
    collectRowsGM_ = workspaceGM_ + off;
    off += cap * HB;
    counterScratchGM_ = workspaceGM_ + off;
    off += static_cast<int64_t>(aiv) * UB_ALIGN;
    srcArriveGM_ = workspaceGM_ + off;
    off += align32(world * 4);
    recvPrefixGM_ = workspaceGM_ + off;
    off += align32(world * 4);
    sendPrefixGM_ = workspaceGM_ + off;
    off += align32((world + 1) * 4);
    flagScratchGM_ = workspaceGM_ + off;
    off += 64; // [0..31]=collectedCount
    segCountGM_ = workspaceGM_ + off;
    off += align32(aiv * 4);
    coreStartGM_ = workspaceGM_ + off;
    off += align32(aiv * 4);
    sortCompanionGM_ = workspaceGM_ + off;
    off += align32(cap * 4);

    constexpr int64_t kRadixRounds = 4;
    int64_t slTileCount = static_cast<int64_t>(t->sortTileCount);
    int64_t seg0 = align32(256 * kRadixRounds * 4);
    int64_t seg1 = align32(slTileCount * 256 * kRadixRounds * 4);
    int64_t seg2 = align32(cap * 4);
    int64_t seg3 = 2 * align32(slTileCount * 256 * 2);
    int64_t seg4 = align32(slTileCount * t->sortNumTileData);
    int64_t seg5 = align32(cap * 4);
    sortWorkspaceGm_ = workspaceGM_ + off;
    off += seg0 + seg1 + seg2 + seg3 + seg4 + seg5;
    softBarrierPageInts_ = aiv * STATE_OFFSET / sizeof(int32_t);
    int64_t sortBarrierBytes = SOFT_BARRIER_MAX_GENS * align32(aiv * STATE_OFFSET);
    sortBarrierGM_ = workspaceGM_ + off;
    off += sortBarrierBytes;
}

__aicore__ inline void EngramFetchGradRailAlignedArch35::InitBuffers(const EngramFetchGradTilingData* t)
{
    tpipe_->InitBuffer(hcommBuf_, HCOMM_INIT_SIZE);
    hcomm_.Init(hcommBuf_.Get<uint8_t>(), HCOMM_INIT_SIZE);

    tpipe_->InitBuffer(creditBuf_, UB_ALIGN);             // 信用读自旋/标量写暂存
    tpipe_->InitBuffer(areaHeadBuf_, RECV_AREA_HEAD);     // 收件区三计数
    uint32_t rowBlockBytes = batchRows_ * win_.rowStride; // 拼行块 ≤64KB
    tpipe_->InitBuffer(rowBlockBuf_, rowBlockBytes);
    tpipe_->InitBuffer(indicesBuf_, IDX_BUF_BYTES); // perm 分片/counts 分块
    indicesBufElems_ = IDX_BUF_BYTES / sizeof(int32_t);
    indicesBufBytes_ = IDX_BUF_BYTES;

    uint32_t statusBytes = static_cast<uint32_t>(
        (2 * STATE_OFFSET + static_cast<uint64_t>(t->aivNum) * 8 + UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN);
    tpipe_->InitBuffer(statusBuf_, statusBytes);
    statusBufBytes_ = statusBytes;
    uint32_t coreArrayBytes =
        static_cast<uint32_t>((((static_cast<uint64_t>(t->aivNum) * 4 + UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN) +
                               static_cast<uint64_t>(t->aivNum) * 4 + UB_ALIGN - 1) /
                              UB_ALIGN * UB_ALIGN);
    uint32_t tempBytes = (statusBytes > ENTRY_BATCH_CAP * 4U) ? statusBytes : ENTRY_BATCH_CAP * 4U;
    if (tempBytes < coreArrayBytes) {
        tempBytes = coreArrayBytes;
    }
    tpipe_->InitBuffer(tempBuf_, tempBytes);
    tempBufBytes_ = tempBytes;
    tpipe_->InitBuffer(entryBuf_, ENTRY_BUF_BYTES); // unsort/行拷贝 ping
    tpipe_->InitBuffer(gradBuf_, GRAD_BUF_BYTES);   // unsort/行拷贝 pong
    uint32_t unsortBatch = 1U;
    if (hiddenBytes_ > 0) {
        uint32_t byTile =
            static_cast<uint32_t>(static_cast<uint64_t>(tileBytes_) / static_cast<uint64_t>(hiddenBytes_));
        unsortBatch = (byTile > 1U) ? byTile : 1U;
        if (unsortBatch > indicesBufElems_) {
            unsortBatch = indicesBufElems_;
        }
    }
    uint32_t offBufBytes = (unsortBatch * sizeof(int64_t) + UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN;
    tpipe_->InitBuffer(offBuf_, offBufBytes);
    // F 组流状态：F 核暂存计数（每槽滚动 staged；索引访问，无指针别名）
    flowBufBytes_ = static_cast<uint32_t>(
        (static_cast<uint64_t>(topo_.ranksPerSupernode) * (sizeof(int32_t)) + UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN);
    tpipe_->InitBuffer(flowBuf_, flowBufBytes_);
    // S 核每目的流状态（统一发送）：5 × 名下目的数 int32（flowW/flowR/taskIdx/ownOff/fin；
    // 目的统一编号 = owner 槽 [0,ranksPerSupernode) + 轨道槽 [ranksPerSupernode,W)，名下目的 = d mod sendCoreCnt ==
    // 本核）
    {
        uint32_t sDests = topo_.ranksPerSupernode + topo_.supernodeCnt;
        uint32_t sCnt = (t->sendCoreCnt > 0U) ? t->sendCoreCnt : 1U;
        uint32_t sFlowInts = 5U * ((sDests + sCnt - 1U) / sCnt);
        tpipe_->InitBuffer(sFlowBuf_,
                           (sFlowInts * static_cast<uint32_t>(sizeof(int32_t)) + UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN);
    }

    // C 阶段缓冲尺寸（与老实现同口径；chunk 模式在 LoadTilingData 已算 uniqueEntryBytes_）
    if (chunkElems_ == 0U) {
        uint32_t inRowStride = static_cast<uint32_t>((hiddenBytes_ + UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN);
        uint32_t maxByPong = Mc2Kernel::GRAD_PING_BYTES / inRowStride;
        if (maxByPong < 1U) {
            maxByPong = 1U;
        }
        if (maxByPong > gradSubBatch_) {
            maxByPong = gradSubBatch_;
        }
        uint32_t fp32RowStride =
            static_cast<uint32_t>((hiddenDim_ * sizeof(float) + UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN);
        if (inputDtype_ != ENGRAM_DT_FLOAT) {
            if (inputDtype_ == outputDtype_) { // list 路径：整 tile 一次性 cast 暂存
                uint32_t listTileRows = Mc2Kernel::GRAD_PING_BYTES / 2U / inRowStride;
                if (listTileRows < 1U) {
                    listTileRows = 1U;
                }
                castBufBytes_ = static_cast<uint32_t>(
                    (static_cast<uint64_t>(hiddenDim_) * sizeof(float) * listTileRows + UB_ALIGN - 1) / UB_ALIGN *
                    UB_ALIGN);
            } else {
                castBufBytes_ = static_cast<uint32_t>(
                    (static_cast<uint64_t>(fp32RowStride) * maxByPong * 2U + UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN);
            }
        } else {
            castBufBytes_ = 0U;
        }
        accumBufBytes_ = static_cast<uint32_t>(
            (static_cast<uint64_t>(fp32RowStride) * ACCUM_BUF_COPIES + UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN);
        uniqueEntryBytes_ = FLUSH_CAST_HEAD_BYTES +
                            ((outputDtype_ != ENGRAM_DT_FLOAT) ?
                                 2U * static_cast<uint32_t>((hiddenDim_ * 2U + UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN) :
                                 0U);
    }

    evMte2ToMte3_[0] = static_cast<int32_t>(tpipe_->FetchEventID(AscendC::HardEvent::MTE2_MTE3));
    evMte2ToMte3_[1] = static_cast<int32_t>(tpipe_->FetchEventID(AscendC::HardEvent::MTE2_MTE3));
    evMte3ToMte2_[0] = static_cast<int32_t>(tpipe_->FetchEventID(AscendC::HardEvent::MTE3_MTE2));
    evMte3ToMte2_[1] = static_cast<int32_t>(tpipe_->FetchEventID(AscendC::HardEvent::MTE3_MTE2));
    evCopy2To3_[0] = static_cast<int32_t>(tpipe_->FetchEventID(AscendC::HardEvent::MTE2_MTE3));
    evCopy2To3_[1] = static_cast<int32_t>(tpipe_->FetchEventID(AscendC::HardEvent::MTE2_MTE3));
    evCopy3To2_[0] = static_cast<int32_t>(tpipe_->FetchEventID(AscendC::HardEvent::MTE3_MTE2));
    evCopy3To2_[1] = static_cast<int32_t>(tpipe_->FetchEventID(AscendC::HardEvent::MTE3_MTE2));
}

// S/F/O 三组（初版 1:2:2；tiling 落表值优先，不一致按比例兜底）
__aicore__ inline void EngramFetchGradRailAlignedArch35::InitCoreGroups()
{
    uint32_t aiv = totalBlocks_;
    if (sendCoreEnd_ > 0U) { // Init 里已按 tiling 落表初始化（见 Init 尾部）
        return;
    }
    // 1:2:2 兜底配比（tiling 落表值非法时）
    uint32_t sCnt = (aiv >= 5U) ? aiv / 5U : 1U;
    uint32_t fCnt = (aiv >= 5U) ? 2U * aiv / 5U : 1U;
    uint32_t oCnt = (aiv > sCnt + fCnt) ? aiv - sCnt - fCnt : 1U;
    if (sCnt + fCnt + oCnt != aiv || sCnt < 1U || fCnt < 1U ||
        oCnt < 1U) { // 极小核数兜底（保 F≥1：relay EOF 发布依赖）
        sCnt = 1U;
        fCnt = 1U;
        oCnt = (aiv > 2U) ? aiv - 2U : 1U;
    }
    sendCoreEnd_ = sCnt;
    fwdCoreEnd_ = sendCoreEnd_ + fCnt;
}

/// @brief kernel 初始化：解析平铺 rail ctx（与 tiling 双源校验，违例 trap）、装载 tiling、
///        初始化通信与 UB 缓冲
__aicore__ inline void EngramFetchGradRailAlignedArch35::Init(GM_ADDR commContext, GM_ADDR gradFetched, GM_ADDR perm,
                                                              GM_ADDR sendCounts, GM_ADDR recvCounts,
                                                              GM_ADDR recvLocalEntry, GM_ADDR numRecv,
                                                              GM_ADDR gradUniqueOut, GM_ADDR uniqueLocalEntryOut,
                                                              GM_ADDR numUniqueOut, GM_ADDR workspaceGM, TPipe* pipe,
                                                              const EngramFetchGradTilingData* tilingData)
{
    tpipe_ = pipe;
    gradFetchedGM_ = gradFetched;
    permGM_ = perm;
    sendCountsGM_ = sendCounts;
    recvCountsGM_ = recvCounts;
    recvLocalEntryGM_ = recvLocalEntry;
    numRecvGM_ = numRecv;
    gradUniqueOutGM_ = gradUniqueOut;
    uniqueLocalEntryOutGM_ = uniqueLocalEntryOut;
    numUniqueOutGM_ = numUniqueOut;
    workspaceGM_ = workspaceGM;
    tileBytes_ = TILE_BYTES;
    aivId_ = AscendC::GetBlockIdx();
    totalBlocks_ = AscendC::GetBlockNum();

    // ═══ 平铺 rail ctx 解析（头部标量 + node 表，纯数据）═══
    ctxHead_ = reinterpret_cast<__gm__ EngramRailCtxHead*>(commContext);
    topo_ = MakeRailTopo(tilingData->ranksPerSupernode, tilingData->rankSize);
    // 双源校验：头部 C == tiling 感知值；头部超节点数 == world/ranksPerSupernode；
    // 本地号/超节点号与全局 rank 对账
    uint32_t myLocal = ctxHead_->localId;
    uint32_t mySn = ctxHead_->supernodeId;
    uint32_t expectCnt = topo_.supernodeCnt;
    ascendc_assert(ctxHead_->ranksPerSupernode == topo_.ranksPerSupernode && ctxHead_->supernodeCnt == expectCnt &&
                       mySn * topo_.ranksPerSupernode + myLocal < tilingData->rankSize,
                   "rail ctx mismatch: C=%u(expect %u), snCnt=%u(expect %u), world=%u", ctxHead_->ranksPerSupernode,
                   topo_.ranksPerSupernode, ctxHead_->supernodeCnt, expectCnt, tilingData->rankSize);
    myGlobalRank_ = mySn * topo_.ranksPerSupernode + myLocal; // rank = sn×C + L（头部标量拼出）
    me_ = DecodeRank(topo_, myGlobalRank_);
    localWin_ = reinterpret_cast<GM_ADDR>(NodeWinAt(reinterpret_cast<GM_ADDR>(ctxHead_), myLocal)); // node 表自槽

    LoadTilingData(tilingData);
    CalcWorkSpaceOffset(tilingData);
    InitBuffers(tilingData);

    // 三组边界：tiling 落表值（sendCoreCnt/fwdCoreCnt/ownerCoreCnt 与 aivNum 一致时优先）
    uint32_t tilingS = tilingData->sendCoreCnt;
    uint32_t tilingF = tilingData->fwdCoreCnt;
    uint32_t tilingO = tilingData->ownerCoreCnt;
    if (tilingS >= 1U && tilingO >= 1U && tilingS + tilingF + tilingO == totalBlocks_) {
        sendCoreEnd_ = tilingS;
        fwdCoreEnd_ = tilingS + tilingF;
    } else {
        sendCoreEnd_ = 0; // 触发 InitCoreGroups 兜底
    }
    uniqueScatter_.Init(aivId_, totalBlocks_, myGlobalRank_, tilingData->rankSize, numEntriesPerRank_, hiddenDim_,
                        hiddenBytes_, inputDtype_, outputDtype_, tpipe_, entryBuf_, gradBuf_, indicesBuf_, tempBuf_,
                        statusBuf_, castFp32Buf_, accumBuf_); // cast/accum 阶段 C 重绑后注入
    uniqueScatter_.SetChunkElems(chunkElems_);
}

// ═══════════════ barrier 两轮协议 ═══════════════
// 窗头 = (W+1)×32B：[0,C) 同伴槽、[C,W) 轨道槽、槽 W 为 epoch（跨调用单调计数）。
// 轮值编码 clr(c)=2c+1 / done(c)=2c+2，单调免清零；两个自槽（myL/mySn）永不置位，跳过。

__aicore__ inline bool EngramFetchGradRailAlignedArch35::IsSelfWriterSlot(uint32_t slot)
{
    const uint32_t ranksPerSupernode = topo_.ranksPerSupernode;
    return slot == me_.local || slot == ranksPerSupernode + me_.supernode;
}

// 槽 → 写者全局 rank（自槽返回 0xFFFFFFFF）
__aicore__ inline uint32_t EngramFetchGradRailAlignedArch35::PeerOfWriterSlot(uint32_t slot)
{
    const uint32_t ranksPerSupernode = topo_.ranksPerSupernode;
    if (slot < ranksPerSupernode) {
        return me_.supernode * ranksPerSupernode + slot; // 同伴
    }
    return (slot - ranksPerSupernode) * ranksPerSupernode + me_.local; // 轨道对端：sn' 超节点的同 L 卡
}

// 轮①：等本卡 W 槽全部 ≥ target（写者们上一轮收尾 flag；首次调用 target=0 直通）
__aicore__ inline void EngramFetchGradRailAlignedArch35::BarrierWaitWritersDone(uint32_t target)
{
    const uint32_t W = topo_.writerSlots;
    AscendC::SyncAll<true>();
    uint64_t t0 = WinNowUs(); // 真墙钟 60s
    for (uint32_t i = aivId_; i < W; i += totalBlocks_) {
        if (IsSelfWriterSlot(i)) {
            continue; // 自槽永不置位
        }
        while (static_cast<uint32_t>(Rd32(localWin_ + static_cast<uint64_t>(i) * STATE_OFFSET)) < target) {
            ascendc_assert(WinNowUs() - t0 < WIN_TIMEOUT_US, "rail barrier round1 timeout: slot=%u, target=%u", i,
                           target);
        }
    }
    AscendC::SyncAll<true>();
}

// 轮②：向全部写者通告"已清可写"（值 target；写者收到才允许 S/F 写我）
__aicore__ inline void EngramFetchGradRailAlignedArch35::BarrierAnnounceCleared(uint32_t target)
{
    // 通告核 = 该写者槽通道的属主 S 核（与数据发送同核，维持每通道单生产者；hcomm 的
    // sqHead 等为普通 GM 标量写，通告与数据分属两核会按陈旧 sqHead 落 WQE 覆盖 SQ 槽）
    const uint32_t W = topo_.writerSlots;
    const uint32_t ranksPerSupernode = topo_.ranksPerSupernode;
    GM_ADDR myScratch = counterScratchGM_ + static_cast<uint64_t>(aivId_) * UB_ALIGN;
    if (aivId_ < sendCoreEnd_) {
        for (uint32_t i = aivId_; i < W; i += sendCoreEnd_) {
            if (IsSelfWriterSlot(i)) {
                continue;
            }
            uint32_t peer = PeerOfWriterSlot(i);
            uint32_t peerL = peer % topo_.ranksPerSupernode;
            uint32_t peerSn = peer / topo_.ranksPerSupernode;
            GM_ADDR ctxBase = reinterpret_cast<GM_ADDR>(ctxHead_);
            uint32_t node = NodeIdxOf(me_.supernode, peerSn, peerL, topo_.ranksPerSupernode);
            uint64_t handle = NodeChannelAt(ctxBase, node);
            GM_ADDR peerWin = reinterpret_cast<GM_ADDR>(NodeWinAt(ctxBase, node));
            // 我在 peer 写者枚举里的槽号（关系对称可算）
            uint32_t mySlot = MySlotInPeer(topo_, me_, peer);
            Wr32(myScratch, static_cast<int32_t>(target));
            int32_t ret = hcomm_.WriteNbi(handle, peerWin + static_cast<uint64_t>(mySlot) * STATE_OFFSET, myScratch,
                                          STATE_OFFSET);
            ascendc_assert(ret == 0, "barrier announce WriteNbi failed, ret=%d, peer=%u", ret, peer);
        }
    }
    AscendC::SyncAll<true>();
}

// 发送门：等 W 槽全部 ≥ target（写者们本轮的"已清可写"通告）
__aicore__ inline void EngramFetchGradRailAlignedArch35::BarrierWaitPeersCleared(uint32_t target)
{
    const uint32_t W = topo_.writerSlots;
    AscendC::SyncAll<true>();
    uint64_t t0 = WinNowUs(); // 真墙钟 60s
    for (uint32_t i = aivId_; i < W; i += totalBlocks_) {
        if (IsSelfWriterSlot(i)) {
            continue;
        }
        while (static_cast<uint32_t>(Rd32(localWin_ + static_cast<uint64_t>(i) * STATE_OFFSET)) < target) {
            ascendc_assert(WinNowUs() - t0 < WIN_TIMEOUT_US, "rail barrier round2 timeout: slot=%u, target=%u", i,
                           target);
        }
    }
    AscendC::SyncAll<true>();
}

// 调用收尾：向全部写者写 done 值 2(c+1)（轮① 的等待目标），核 0 推 epoch 槽
__aicore__ inline void EngramFetchGradRailAlignedArch35::BarrierWriteCallDone(uint32_t doneVal)
{
    // 通告核 = 通道属主 S 核（同 BarrierAnnounceCleared：全调用内每通道恒单生产者）
    const uint32_t W = topo_.writerSlots;
    GM_ADDR myScratch = counterScratchGM_ + static_cast<uint64_t>(aivId_) * UB_ALIGN;
    if (aivId_ < sendCoreEnd_) {
        for (uint32_t i = aivId_; i < W; i += sendCoreEnd_) {
            if (IsSelfWriterSlot(i)) {
                continue;
            }
            uint32_t peer = PeerOfWriterSlot(i);
            uint32_t peerL = peer % topo_.ranksPerSupernode;
            uint32_t peerSn = peer / topo_.ranksPerSupernode;
            GM_ADDR ctxBase = reinterpret_cast<GM_ADDR>(ctxHead_);
            uint32_t node = NodeIdxOf(me_.supernode, peerSn, peerL, topo_.ranksPerSupernode);
            uint64_t handle = NodeChannelAt(ctxBase, node);
            GM_ADDR peerWin = reinterpret_cast<GM_ADDR>(NodeWinAt(ctxBase, node));
            uint32_t mySlot = MySlotInPeer(topo_, me_, peer);
            Wr32(myScratch, static_cast<int32_t>(doneVal));
            int32_t ret = hcomm_.WriteNbi(handle, peerWin + static_cast<uint64_t>(mySlot) * STATE_OFFSET, myScratch,
                                          STATE_OFFSET);
            ascendc_assert(ret == 0, "barrier done WriteNbi failed, ret=%d, peer=%u", ret, peer);
        }
    }
    AscendC::SyncAll<true>();
    if (aivId_ == 0) { // epoch 槽推进（下一调用轮① 的等待目标 = 本 done 值）
        Wr32(localWin_ + static_cast<uint64_t>(topo_.writerSlots) * STATE_OFFSET, static_cast<int32_t>(doneVal));
    }
    AscendC::SyncAll<true>();
}

// ═══════════════ 阶段 A ═══════════════

// GM 段清零（[r0, r1) 个 int32，本核分片；Duplicate → DataCopy 分块——DataCopyPad 兼容非对齐尾块）
__aicore__ inline void EngramFetchGradRailAlignedArch35::ClearGmRange(GM_ADDR base, int64_t r0, int64_t r1)
{
    if (r1 <= r0) {
        return;
    }
    LocalTensor<int32_t> zero = tempBuf_.Get<int32_t>();
    AscendC::Duplicate<int32_t>(zero, 0, ENTRY_BATCH_CAP);
    SyncEv<AscendC::HardEvent::V_MTE3>();
    int64_t done = r0;
    while (done < r1) {
        int64_t n = r1 - done;
        if (n > ENTRY_BATCH_CAP) {
            n = ENTRY_BATCH_CAP;
        }
        GlobalTensor<int32_t> rangeGM;
        rangeGM.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(base + done * 4));
        // DataCopyPad：缩比床 world/段数 < 批容量时尾块 n*4 非 32B 对齐——裸 DataCopy 会整块丢弃
        AscendC::DataCopyParams cp{1U, static_cast<uint16_t>(n * 4), 0U, 0U};
        AscendC::DataCopyPad(rangeGM, zero, cp);
        done += n;
    }
    SyncEv<AscendC::HardEvent::MTE3_S>();
}

__aicore__ inline void EngramFetchGradRailAlignedArch35::ClearRecvAreas()
{
    // 全核分片清 W 个收件区（rail: world/ranksPerSupernode 格 + owner: ranksPerSupernode 格）96B 头
    // + srcArrive(world×4B) + doneCtl（S+ranksPerSupernode+3 项）
    const uint32_t ranksPerSupernode = topo_.ranksPerSupernode;
    const uint64_t totalAreas = static_cast<uint64_t>(topo_.supernodeCnt) + ranksPerSupernode;
    LocalTensor<int32_t> zero = tempBuf_.Get<int32_t>();
    AscendC::Duplicate<int32_t>(zero, 0, RECV_AREA_HEAD / sizeof(int32_t));
    SyncEv<AscendC::HardEvent::V_MTE3>();
    for (uint64_t i = aivId_; i < totalAreas; i += totalBlocks_) {
        GM_ADDR area = nullptr;
        if (i < topo_.supernodeCnt) {
            area = RailArea(localWin_, win_, static_cast<uint32_t>(i));
        } else {
            area = OwnArea(localWin_, win_, static_cast<uint32_t>(i - topo_.supernodeCnt));
        }
        GlobalTensor<int32_t> headGM;
        headGM.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(area));
        AscendC::DataCopyParams headCp{1U, static_cast<uint16_t>(RECV_AREA_HEAD), 0U, 0U};
        AscendC::DataCopyPad(headGM, zero, headCp);
    }
    // srcArrive 分片清零（world×4B）
    int64_t world = static_cast<int64_t>(topo_.world);
    int64_t per = (world + totalBlocks_ - 1) / totalBlocks_;
    int64_t s0 = static_cast<int64_t>(aivId_) * per;
    int64_t s1 = (s0 + per > world) ? world : s0 + per;
    ClearGmRange(srcArriveGM_, s0, s1);
    const uint32_t fCnt = (fwdCoreEnd_ > sendCoreEnd_) ? (fwdCoreEnd_ - sendCoreEnd_) : 1U;
    const int64_t rawInts = static_cast<int64_t>(fCnt) * static_cast<int64_t>(ranksPerSupernode);
    const int64_t alignInts = (rawInts * 4 + UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN / 4;
    const int64_t ctlCnt = 3 * alignInts;
    per = (ctlCnt + totalBlocks_ - 1) / totalBlocks_;
    s0 = static_cast<int64_t>(aivId_) * per;
    s1 = (s0 + per > ctlCnt) ? ctlCnt : s0 + per;
    ClearGmRange(relayStagedGM_, s0, s1);
    if (aivId_ == 0) {
        // flagScratch 全部标量槽清零：[0]=collectedCount 汇总，[1]=S ΣwriteCnt，[2]=F Σstaged（DFX AtomicAdd 目标）
        for (uint32_t w = 0; w < DFX_SCALAR_SLOTS; ++w) {
            Wr32(flagScratchGM_ + w * 4, 0);
        }
    }
}

// 并行两遍前缀（segCount/coreStart 作 scratch）：counts[p] 源（int32 紧排），prefix[r] 目的
__aicore__ inline void EngramFetchGradRailAlignedArch35::BuildRecvPrefix()
{
    int64_t world = static_cast<int64_t>(topo_.world);
    int64_t per = (world + totalBlocks_ - 1) / totalBlocks_;
    int64_t r0 = static_cast<int64_t>(aivId_) * per;
    int64_t r1 = (r0 + per > world) ? world : r0 + per;
    if (r0 >= world) {
        r0 = world; // 空段不可早退：SyncAll 要求全核到达，早退会让其余核永久等待
        r1 = world;
    }
    // pass1：段内和
    int64_t segSum = 0;
    for (int64_t r = r0; r < r1; ++r) {
        segSum += Rd32(recvCountsGM_ + r * 4);
    }
    Wr32(segCountGM_ + aivId_ * 4, static_cast<int32_t>(segSum)); // 空段写 0（核0 pass2 读全核段和）
    AscendC::SyncAll<true>();
    // pass2：核0 算段偏移
    if (aivId_ == 0) {
        int64_t base = 0;
        for (uint32_t c = 0; c < totalBlocks_; ++c) {
            Wr32(coreStartGM_ + c * 4, static_cast<int32_t>(base));
            base += Rd32(segCountGM_ + c * 4);
        }
    }
    AscendC::SyncAll<true>();
    // pass3：段内前缀落表
    int64_t base = Rd32(coreStartGM_ + aivId_ * 4);
    // 逐条 Wr32：连续复用同一 creditBuf_ 时需尾部反向同步，防 MTE3 积压下源字未读走致前缀错表
    for (int64_t r = r0; r < r1; ++r) {
        Wr32(recvPrefixGM_ + r * 4, static_cast<int32_t>(base));
        base += Rd32(recvCountsGM_ + r * 4);
    }
    // 不补写 recvPrefix[world]：该位无读者（numRecv 从前向输出读取），越段补写会覆盖紧邻段首
}

// 同 BuildRecvPrefix，源换 sendCounts（32B 槽距），含 sendPrefix[world]=T
__aicore__ inline void EngramFetchGradRailAlignedArch35::BuildSendPrefix()
{
    int64_t world = static_cast<int64_t>(topo_.world);
    int64_t per = (world + totalBlocks_ - 1) / totalBlocks_;
    int64_t r0 = static_cast<int64_t>(aivId_) * per;
    int64_t r1 = (r0 + per > world) ? world : r0 + per;
    if (r0 >= world) {
        // 空段不可早退（同 BuildRecvPrefix：SyncAll 要求全核到达）
        r0 = world;
        r1 = world;
    }
    int64_t segSum = 0;
    for (int64_t r = r0; r < r1; ++r) {
        segSum += Rd32(sendCountsGM_ + r * 32); // sendCounts 每项 32B 槽距
    }
    Wr32(segCountGM_ + aivId_ * 4, static_cast<int32_t>(segSum));
    AscendC::SyncAll<true>();
    if (aivId_ == 0) {
        int64_t base = 0;
        for (uint32_t c = 0; c < totalBlocks_; ++c) {
            Wr32(coreStartGM_ + c * 4, static_cast<int32_t>(base));
            base += Rd32(segCountGM_ + c * 4);
        }
    }
    AscendC::SyncAll<true>();
    int64_t base = Rd32(coreStartGM_ + aivId_ * 4);
    // 逐条 Wr32：连续复用同一 creditBuf_ 时需尾部反向同步，防 MTE3 积压下源字未读走致前缀错表
    for (int64_t r = r0; r < r1; ++r) {
        Wr32(sendPrefixGM_ + r * 4, static_cast<int32_t>(base));
        base += Rd32(sendCountsGM_ + r * 32);
    }
    if (r0 < world && r1 == world) { // 末段核补 T 总量（r0<world 防空段核误写）
        Wr32(sendPrefixGM_ + world * 4, static_cast<int32_t>(base));
    }
}

// 每行 dst 伴随：二分 sendPrefix + 区间 Duplicate 填充（替代老 O(rankSize) 标量前缀和）
__aicore__ inline void EngramFetchGradRailAlignedArch35::MarkOwnerSorted()
{
    int64_t per = (numTokens_ + totalBlocks_ - 1) / totalBlocks_;
    int64_t start = static_cast<int64_t>(aivId_) * per;
    int64_t end = (start + per > numTokens_) ? numTokens_ : start + per;
    if (start >= end) {
        return;
    }
    int64_t world = static_cast<int64_t>(topo_.world);
    LocalTensor<int32_t> fill = tempBuf_.Get<int32_t>();
    int64_t row = start;
    while (row < end) {
        // 二分：最大 r 使 sendPrefix[r] <= row
        int64_t lo = 0;
        int64_t hi = world - 1;
        while (lo < hi) {
            int64_t mid = (lo + hi + 1) / 2;
            if (static_cast<int64_t>(Rd32(sendPrefixGM_ + mid * 4)) <= row) {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        int64_t dst = lo;
        int64_t segEnd = Rd32(sendPrefixGM_ + (dst + 1) * 4);
        if (segEnd > end) {
            segEnd = end;
        }
        int64_t n = segEnd - row;
        // 分块填充 + DataCopyPad：兼容非 32B 对齐长度/地址，且 Duplicate 长度受 tempBuf 容量约束
        GlobalTensor<int32_t> outGM;
        outGM.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(ownerSortedGM_ + row * 4));
        int64_t done = 0;
        while (done < n) {
            int64_t m = n - done;
            if (m > static_cast<int64_t>(ENTRY_BATCH_CAP)) {
                m = ENTRY_BATCH_CAP;
            }
            AscendC::Duplicate<int32_t>(fill, static_cast<int32_t>(dst), static_cast<int32_t>(m));
            SyncEv<AscendC::HardEvent::V_MTE3>();
            AscendC::DataCopyParams cp{1U, static_cast<uint16_t>(m * 4), 0U, 0U};
            AscendC::DataCopyPad(outGM[static_cast<uint32_t>(done)], fill, cp);
            SyncEv<AscendC::HardEvent::MTE3_S>();
            done += m;
        }
        row = segEnd;
    }
}

// 四桶段表：每核扫自己切片 → 本核条带（dst 升序保证行天然连续段，段数 ≤ 行/核+1）
__aicore__ inline void EngramFetchGradRailAlignedArch35::BucketClassify()
{
    int64_t per = (numTokens_ + totalBlocks_ - 1) / totalBlocks_;
    int64_t start = static_cast<int64_t>(aivId_) * per;
    int64_t end = (start + per > numTokens_) ? numTokens_ : start + per;
    if (start >= end) {
        Wr32(segCountsGM_ + (1 + aivId_) * 4, 0);
        return;
    }
    GM_ADDR strip = stripsGM_ + static_cast<uint64_t>(aivId_) * stripCap_ * 12;
    GlobalTensor<int32_t> ownGM;
    ownGM.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(ownerSortedGM_));
    LocalTensor<int32_t> chunk = indicesBuf_.Get<int32_t>();
    uint32_t runCnt = 0;
    int64_t pos = start;
    int32_t curDst = -1;
    int64_t runStart = start;
    int64_t runLen = 0;
    // 条带落表必须走 Wr32：裸标量存 GM 只进本核 L1 写回缓存，SyncAll 不刷，合并核读不到
    auto emitRun = [&]() __attribute__((cce_aicore))
    {
        Wr32(strip + runCnt * 12 + 0, curDst);
        Wr32(strip + runCnt * 12 + 4, static_cast<int32_t>(runStart));
        Wr32(strip + runCnt * 12 + 8, static_cast<int32_t>(runLen));
        runCnt++;
    };
    while (pos < end) {
        int64_t n = end - pos;
        if (n > indicesBufElems_) {
            n = indicesBufElems_;
        }
        AscendC::DataCopyExtParams cp{1U, static_cast<uint32_t>(n * 4), 0U, 0U, 0U};
        AscendC::DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
        AscendC::DataCopyPad(chunk, ownGM[static_cast<uint32_t>(pos)], cp, pad);
        SyncEv<AscendC::HardEvent::MTE2_S>();
        for (int64_t j = 0; j < n; ++j) {
            int32_t dst = chunk.GetValue(static_cast<int32_t>(j));
            if (dst != curDst) {
                // dst 升序不变量（每收件区恰一任务的前提）：回退即 fail-fast，防静默丢行
                ascendc_assert(curDst < 0 || dst > curDst,
                               "rail ownerSorted not ascending: core=%u row=%lld dst=%d prev=%d", aivId_,
                               static_cast<int64_t>(pos + j), dst, curDst);
                if (curDst >= 0) { // 上一段出条带
                    emitRun();
                }
                curDst = dst;
                runStart = pos + j;
                runLen = 0;
            }
            runLen++;
        }
        pos += n;
    }
    if (curDst >= 0) {
        emitRun();
    }
    Wr32(segCountsGM_ + (1 + aivId_) * 4, static_cast<int32_t>(runCnt));
}

// S0 合并：同 dst 相邻段合并为单一任务（write/done 计数为每 (src→dst) 流唯一，双写互相覆盖）
// 任务表项 (kind, dstRank, startRow, rowCount)；LOCAL 任务保留（O 组取数，S 组跳过）
__aicore__ inline void EngramFetchGradRailAlignedArch35::MergeSegmentsBySGroup()
{
    if (aivId_ != 0) {
        return;
    }
    auto rdStrip = [&](uint32_t core, uint32_t i) __attribute__((cce_aicore))
    {
        return stripsGM_ + static_cast<uint64_t>(core) * stripCap_ * 12 + i * 12;
    };
    int32_t curDst = -1;
    int64_t runStart = 0;
    int64_t runLen = 0;
    uint32_t taskCnt = 0;
    // 任务按"路由收件区"合并（DIRECT 键=dst 卡、RAIL 键=dst 超节点号）：每目的一条流，一区一任务
    auto areaKey = [&](int32_t dstRank) __attribute__((cce_aicore))->int32_t
    {
        RankCoord c = DecodeRank(topo_, static_cast<uint32_t>(dstRank));
        BucketKind k = ClassifyRow(topo_, me_, static_cast<uint32_t>(dstRank));
        int32_t sub = (k == BUCKET_DIRECT) ? static_cast<int32_t>(c.local) : static_cast<int32_t>(c.supernode);
        return static_cast<int32_t>(k) * RAIL_BUCKET_KEY_STRIDE + sub;
    };
    // 任务表写入必须 Wr32（UB→MTE3 DataCopy）：裸标量存只进本核 L1（写回），SyncAll 不刷
    // 数据缓存——S/O 组读到陈旧零 → cnt=0 秒完结 → 行全部静默丢失
    auto emit = [&]() __attribute__((cce_aicore))
    {
        if (curDst < 0) {
            return;
        }
        BucketKind kind = ClassifyRow(topo_, me_, static_cast<uint32_t>(curDst));
        Wr32(taskTableGM_ + taskCnt * 16 + 0, static_cast<int32_t>(kind));
        Wr32(taskTableGM_ + taskCnt * 16 + 4, curDst);
        Wr32(taskTableGM_ + taskCnt * 16 + 8, static_cast<int32_t>(runStart));
        Wr32(taskTableGM_ + taskCnt * 16 + 12, static_cast<int32_t>(runLen));
        taskCnt++;
    };
    for (uint32_t c = 0; c < totalBlocks_; ++c) {
        int32_t cnt = Rd32(segCountsGM_ + (1 + c) * 4);
        for (int32_t i = 0; i < cnt; ++i) {
            GM_ADDR seg = rdStrip(c, static_cast<uint32_t>(i)); // bypass 读（跨核可见性同因）
            int32_t dst = Rd32(seg);
            int32_t st = Rd32(seg + 4);
            int32_t len = Rd32(seg + 8);
            if (curDst >= 0 && areaKey(dst) == areaKey(curDst) && static_cast<int64_t>(st) == runStart + runLen) {
                runLen += len; // 同收件区且行连续 → 拼接为一个任务
            } else {
                emit();
                curDst = dst;
                runStart = st;
                runLen = len;
            }
        }
    }
    emit();
    Wr32(segCountsGM_, static_cast<int32_t>(taskCnt));
}

// ═══════════════ 阶段 A：Unsort（语义同老 :616，LocalCopy 换 WinLocalCopy） ═══════════════

__aicore__ inline uint32_t EngramFetchGradRailAlignedArch35::LoadGradChunk(int64_t pos, int64_t end,
                                                                           LocalTensor<uint8_t>& buf, int32_t bufIdx,
                                                                           LocalTensor<int32_t>& idxUb,
                                                                           uint32_t tokensPerBuf)
{
    uint32_t n = 0;
    if (end > pos) {
        n = static_cast<uint32_t>(end - pos);
        if (n > tokensPerBuf) {
            n = tokensPerBuf;
        }
    }
    if (n == 0) {
        return 0;
    }
    GlobalTensor<int32_t> permGM;
    permGM.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(permGM_));
    AscendC::DataCopyExtParams cpParams{1U, static_cast<uint32_t>(n * sizeof(int32_t)), 0U, 0U, 0U};
    AscendC::DataCopyPadExtParams<int32_t> cpPad{false, 0, 0, 0};
    AscendC::DataCopyPad(idxUb, permGM[static_cast<uint32_t>(pos)], cpParams, cpPad);
    SyncEv<AscendC::HardEvent::MTE2_S>();

    GlobalTensor<uint8_t> srcGm;
    srcGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(gradFetchedGM_));
    AscendC::DataCopyPadExtParams<uint8_t> gPad{false, 0, 0, 0};
    uint32_t hiddenBytes32 = static_cast<uint32_t>(hiddenBytes_);
    LocalTensor<int64_t> offUb = offBuf_.Get<int64_t>();
    AscendC::Cast<int64_t, int32_t>(offUb, idxUb, AscendC::RoundMode::CAST_NONE, static_cast<int32_t>(n));
    AscendC::Muls<int64_t>(offUb, offUb, hiddenBytes_, static_cast<int32_t>(n));
    SyncEv<AscendC::HardEvent::V_S>();
    for (uint32_t j = 0; j < n; ++j) {
        int64_t srcOff = offUb.GetValue(j);
        AscendC::DataCopyExtParams gParams{1U, hiddenBytes32, 0U, 0U, 0U};
        AscendC::DataCopyPad(buf[static_cast<uint64_t>(j) * hiddenBytes_], srcGm[srcOff], gParams, gPad);
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(evMte2ToMte3_[bufIdx]);
    return n;
}

__aicore__ inline void EngramFetchGradRailAlignedArch35::StoreGradChunk(int64_t base, uint32_t count,
                                                                        LocalTensor<uint8_t>& buf, int32_t bufIdx)
{
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(evMte2ToMte3_[bufIdx]);
    GlobalTensor<uint8_t> dstGm;
    dstGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(gradSortedGM_));
    uint64_t dstOff = static_cast<uint64_t>(base) * static_cast<uint64_t>(hiddenBytes_);
    uint64_t totalBytes = static_cast<uint64_t>(count) * static_cast<uint64_t>(hiddenBytes_);
    uint64_t srcOff = 0;
    while (totalBytes > 0) {
        uint32_t chunkBytes = (totalBytes > MAX_BLOCK_BYTES) ? MAX_BLOCK_BYTES : static_cast<uint32_t>(totalBytes);
        AscendC::DataCopyExtParams sParams{1U, chunkBytes, 0U, 0U, 0U};
        AscendC::DataCopyPad(dstGm[dstOff + srcOff], buf[srcOff], sParams);
        srcOff += chunkBytes;
        totalBytes -= chunkBytes;
    }
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(evMte3ToMte2_[bufIdx]);
}

__aicore__ inline void EngramFetchGradRailAlignedArch35::UnsortGradRail()
{
    int64_t totalPerCore = (numTokens_ + static_cast<int64_t>(totalBlocks_) - 1) / totalBlocks_;
    int64_t start = static_cast<int64_t>(aivId_) * totalPerCore;
    int64_t end = start + totalPerCore;
    if (end > numTokens_) {
        end = numTokens_;
    }
    if (start >= end) {
        return;
    }

    // 非 32B 对齐 / 超 tile 行：run-length 合并 + WinLocalCopy 逐段
    if ((static_cast<uint64_t>(hiddenBytes_) % UB_ALIGN != 0U) || (static_cast<uint64_t>(hiddenBytes_) > tileBytes_)) {
        LocalTensor<int32_t> idxUb = indicesBuf_.Get<int32_t>();
        int64_t pos = start;
        while (pos < end) {
            uint32_t batch = static_cast<uint32_t>(end - pos);
            if (batch > indicesBufElems_) {
                batch = indicesBufElems_;
            }
            GlobalTensor<int32_t> permGM;
            permGM.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(permGM_));
            AscendC::DataCopyExtParams cpParams{1U, static_cast<uint32_t>(batch * sizeof(int32_t)), 0U, 0U, 0U};
            AscendC::DataCopyPadExtParams<int32_t> cpPad{false, 0, 0, 0};
            AscendC::DataCopyPad(idxUb, permGM[static_cast<uint32_t>(pos)], cpParams, cpPad);
            SyncEv<AscendC::HardEvent::MTE2_S>();
            uint32_t j = 0;
            while (j < batch) {
                int32_t runStartIdx = idxUb.GetValue(static_cast<int32_t>(j));
                uint32_t runLen = 1;
                while (j + runLen < batch) {
                    int32_t nextIdx = idxUb.GetValue(static_cast<int32_t>(j + runLen));
                    if (nextIdx == runStartIdx + static_cast<int32_t>(runLen)) {
                        runLen++;
                    } else {
                        break;
                    }
                }
                GM_ADDR src = gradFetchedGM_ + static_cast<uint64_t>(runStartIdx) * hiddenBytes_;
                GM_ADDR dst = gradSortedGM_ + static_cast<uint64_t>(pos + j) * hiddenBytes_;
                WinLocalCopy(dst, src, static_cast<uint64_t>(runLen) * hiddenBytes_, tpipe_, entryBuf_.Get<uint8_t>(),
                             gradBuf_.Get<uint8_t>(), evCopy2To3_, evCopy3To2_);
                j += runLen;
            }
            pos += batch;
        }
        return;
    }

    // 对齐路径：Load/Store 乒乓（语义同老）
    uint32_t tokensPerBuf = 1U;
    uint32_t byTile = tileBytes_ / static_cast<uint32_t>(hiddenBytes_);
    tokensPerBuf = (byTile > 1U) ? byTile : 1U;
    if (tokensPerBuf > indicesBufElems_) {
        tokensPerBuf = indicesBufElems_;
    }
    LocalTensor<uint8_t> buf0 = entryBuf_.Get<uint8_t>();
    LocalTensor<uint8_t> buf1 = gradBuf_.Get<uint8_t>();
    LocalTensor<int32_t> idxUb = indicesBuf_.Get<int32_t>();
    int32_t curBuf = 0;
    int64_t pos = start;
    int64_t base0 = start;
    uint32_t n0 = LoadGradChunk(pos, end, buf0, curBuf, idxUb, tokensPerBuf);
    pos += n0;
    bool stored0 = false;
    bool stored1 = false;
    while (pos < end) {
        int32_t nxtBuf = 1 - curBuf;
        if (stored0 && nxtBuf == 0) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(evMte3ToMte2_[0]);
        }
        if (stored1 && nxtBuf == 1) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(evMte3ToMte2_[1]);
        }
        uint32_t n1 = LoadGradChunk(pos, end, (nxtBuf == 0) ? buf0 : buf1, nxtBuf, idxUb, tokensPerBuf);
        pos += n1;
        StoreGradChunk(base0, n0, (curBuf == 0) ? buf0 : buf1, curBuf);
        (curBuf == 0) ? (stored0 = true) : (stored1 = true);
        base0 += n0;
        curBuf = nxtBuf;
        n0 = n1;
    }
    StoreGradChunk(base0, n0, (curBuf == 0) ? buf0 : buf1, curBuf);
    (curBuf == 0) ? (stored0 = true) : (stored1 = true);
    if (stored0) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(evMte3ToMte2_[0]);
    }
    if (stored1) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(evMte3ToMte2_[1]);
    }
}

// ═══════════════ 阶段 B ═══════════════

// ── S 核统一发送：目的 → 属主 S 核固定映射（d ∈ [0, dests)：d<C 为 owner 同伴槽，
//    其余为轨道槽；属主 = d mod sendCoreCnt），该对端全部远程写同核同流（每通道单生产者）。
//    轮转推进，信用满/暂存空的目的让位；每批后 Drain（WriteNbi 非阻塞取源）。
/// @brief S 核统一发送循环：每目的一条流，自有桶行 + F 暂存转发行；完结后写 done
/// @return 无（aivId_ 不在 S 组时直接返回）
__aicore__ inline void EngramFetchGradRailAlignedArch35::SendLoopS()
{
    if (aivId_ >= sendCoreEnd_) {
        Wr32(segCountGM_ + aivId_ * 4, 0); // S 核不收集（Σ 交叉校验覆盖全核）
        return;
    }
    const uint32_t sCoreIdx = aivId_;
    const uint32_t sCoreCnt = sendCoreEnd_;
    const uint32_t ranksPerSupernode = topo_.ranksPerSupernode;
    const uint32_t dests = topo_.writerSlots; // ranksPerSupernode + world/ranksPerSupernode（owner 槽 + 轨道槽）
    const uint32_t fCoreCnt = fwdCoreEnd_ - sendCoreEnd_;
    // 名下目的数：d mod sCoreCnt == 本核（按核精确计数，防高编号核算出越界 dIdx）
    const uint32_t myDests = (sCoreIdx < dests) ? ((dests - 1U - sCoreIdx) / sCoreCnt + 1U) : 0U;
    int32_t taskCnt = Rd32(segCountsGM_);
    auto prog = [&](int32_t code) __attribute__((cce_aicore))
    {
        Wr32(segCountGM_ + static_cast<uint64_t>(aivId_) * 4, code); // DFX 进度（终值写 0）
    };
    prog(10); // 进 B 阶段

    // 本核目的状态（UB 紧凑布局，5 段各 myDests 项）：flowW | flowR | taskIdx(−1 无任务，
    // 0 是合法任务号不可作哨兵) | ownOff | fin
    LocalTensor<int32_t> sflow = sFlowBuf_.Get<int32_t>();
    for (uint32_t i = 0; i < 5U * myDests; ++i) {
        sflow.SetValue(static_cast<int32_t>(i), 0);
    }
    for (uint32_t k = 0; k < myDests; ++k) {
        sflow.SetValue(static_cast<int32_t>(2 * myDests + k), -1); // 无任务显式 −1
    }
    uint32_t unfinished = 0;
    for (uint32_t k = 0; k < myDests; ++k) {
        uint32_t dIdx = sCoreIdx + k * sCoreCnt;
        bool selfSlot = (dIdx < ranksPerSupernode && dIdx == me_.local) ||
                        (dIdx >= ranksPerSupernode && dIdx == ranksPerSupernode + me_.supernode);
        if (selfSlot) {
            sflow.SetValue(static_cast<int32_t>(4 * myDests + k), 1); // 自槽空置——永不完结工作
        } else {
            unfinished++;
        }
    }
    // 扫任务表一遍：任务 → 统一目的编号 → 属主核登记（任务按收件区合并，每目的 ≤1 任务；
    // 任务表读取必须 Rd32 bypass（裸标量读命中本核 L1 陈旧行）
    for (int32_t taskIdx = 0; taskIdx < taskCnt; ++taskIdx) {
        GM_ADDR taskG = taskTableGM_ + static_cast<uint64_t>(taskIdx) * 16;
        int32_t kindV = Rd32(taskG);
        uint32_t dst = static_cast<uint32_t>(Rd32(taskG + 4));
        RankCoord d = DecodeRank(topo_, dst);
        uint32_t dIdx = 0;
        if (static_cast<BucketKind>(kindV) == BUCKET_DIRECT) {
            dIdx = d.local; // 同伴卡 → owner 槽
        } else {
            dIdx = ranksPerSupernode + d.supernode; // 轨道槽（dst 超节点的同 L 卡；LOCAL 落自槽，跳过）
        }
        // 防御：dst 非法时 dIdx 越界，越界写会腐蚀 sflow 邻域 UB 且行静默丢失
        if (dIdx < dests && dIdx % sCoreCnt == sCoreIdx) {
            // 同区第二任务 = 条带碎片化：覆盖注册会静默丢行（本地桶例外，O 组直读全表）
            int32_t regSlot = static_cast<int32_t>(2 * myDests + dIdx / sCoreCnt);
            int32_t prev = sflow.GetValue(regSlot);
            ascendc_assert(prev < 0 || static_cast<BucketKind>(kindV) == BUCKET_LOCAL,
                           "rail fragmented tasks on dest %u: task=%d shadows prev=%d, rows lost", dIdx, taskIdx, prev);
            sflow.SetValue(regSlot, taskIdx);
        }
    }

    // 目的 → 通道句柄/对端收件区（算术直出；自槽不可达——映射排除；node 表 = S 核目的编号 d 同表直查）
    auto destAddr = [&](uint32_t dIdx, uint64_t & handle, GM_ADDR & area) __attribute__((cce_aicore))
    {
        GM_ADDR ctxBase = reinterpret_cast<GM_ADDR>(ctxHead_);
        handle = NodeChannelAt(ctxBase, dIdx);
        if (dIdx < ranksPerSupernode) { // owner 槽 L'：写同伴 owner-区[myL]
            area = reinterpret_cast<GM_ADDR>(NodeWinAt(ctxBase, dIdx)) + win_.ownBase +
                   static_cast<uint64_t>(me_.local) * win_.areaSize;
        } else { // 轨道槽 sn'：写轨道对端（sn' 超节点的同 L 卡）rail-区[mySn]
            area = reinterpret_cast<GM_ADDR>(NodeWinAt(ctxBase, dIdx)) + win_.railBase +
                   static_cast<uint64_t>(me_.supernode) * win_.areaSize;
        }
    };

    LocalTensor<uint8_t> block = rowBlockBuf_.Get<uint8_t>();
    LocalTensor<int32_t> head = rowBlockBuf_.Get<int32_t>();
    uint64_t idleSpin = 0;        // 连续无进展轮数（DFX 用）
    uint64_t idleT0 = WinNowUs(); // 距上次进展真墙钟（真 60s 兜底）
    while (unfinished > 0U) {
        bool progressed = false;
        for (uint32_t k = 0; k < myDests; ++k) {
            if (sflow.GetValue(static_cast<int32_t>(4 * myDests + k)) != 0) {
                continue;
            }
            uint32_t dIdx = sCoreIdx + k * sCoreCnt;
            uint64_t handle = 0;
            GM_ADDR area = nullptr;
            destAddr(dIdx, handle, area);
            RowSender sender;
            sender.Init(hcomm_, counterScratchGM_ + aivId_ * UB_ALIGN, tpipe_, creditBuf_.Get<int32_t>(), win_, aivId_,
                        myGlobalRank_, static_cast<uint32_t>(sflow.GetValue(static_cast<int32_t>(k))),
                        sflow.GetValue(static_cast<int32_t>(myDests + k)));
            bool sent = false;
            // ── ① 自有桶行（gradSorted 数据 + ownerSorted 伴随头 → sStage → 远程写）──
            // 任务表读取必须 Rd32 bypass（裸标量读命中本核 L1 陈旧行）
            int32_t taskIdx = sflow.GetValue(static_cast<int32_t>(2 * myDests + k));
            int64_t taskStart = 0;
            int64_t taskRows = 0;
            if (taskIdx >= 0) {
                GM_ADDR taskG = taskTableGM_ + static_cast<uint64_t>(taskIdx) * 16;
                taskStart = Rd32(taskG + 8);
                taskRows = Rd32(taskG + 12);
            }
            if (taskIdx >= 0) {
                int64_t off = sflow.GetValue(static_cast<int32_t>(3 * myDests + k));
                if (off < taskRows) {
                    uint32_t rows = static_cast<uint32_t>((taskRows - off > batchRows_) ? batchRows_ : taskRows - off);
                    if (!sender.HasCredit(rows)) {
                        sender.AskCredit(handle, area); // 异步信用读——先转其它目的
                    } else {
                        uint32_t rowStrideWords = win_.rowStride / 4; // 行距按 int32 字数（行头填充用）
                        for (uint32_t j = 0; j < rows; ++j) {
                            int32_t rowDst = Rd32(ownerSortedGM_ + static_cast<uint64_t>(taskStart + off + j) * 4);
                            head.SetValue(static_cast<int32_t>(j * rowStrideWords), rowDst);
                            head.SetValue(static_cast<int32_t>(j * rowStrideWords + 1),
                                          static_cast<int32_t>(myGlobalRank_));
                        }
                        SyncEv<AscendC::HardEvent::S_MTE2>();
                        GlobalTensor<uint8_t> srcGm;
                        srcGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(gradSortedGM_));
                        AscendC::DataCopyPadExtParams<uint8_t> pad{false, 0, 0, 0};
                        for (uint32_t j = 0; j < rows; ++j) {
                            AscendC::DataCopyExtParams gParams{1U, static_cast<uint32_t>(hiddenBytes_), 0U, 0U, 0U};
                            AscendC::DataCopyPad(block[static_cast<uint64_t>(j) * win_.rowStride + ROW_HEAD_BYTES],
                                                 srcGm[static_cast<uint64_t>(taskStart + off + j) * hiddenBytes_],
                                                 gParams, pad);
                        }
                        SyncEv<AscendC::HardEvent::MTE2_MTE3>();
                        SyncEv<AscendC::HardEvent::S_MTE3>();
                        GM_ADDR stage = sStageGM_ + static_cast<uint64_t>(sCoreIdx) * batchRows_ * win_.rowStride;
                        GlobalTensor<uint8_t> stageGm;
                        stageGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(stage));
                        AscendC::DataCopyExtParams sParams{1U, static_cast<uint32_t>(rows) * win_.rowStride, 0U, 0U,
                                                           0U};
                        AscendC::DataCopyPad(stageGm, block, sParams);
                        SyncEv<AscendC::HardEvent::MTE3_S>();
                        sender.SendBatch(handle, area, stage, rows);
                        sender.DrainChannel(handle); // 通道取走 stage 后才可复写（下一目的拼块）
                        sflow.SetValue(static_cast<int32_t>(3 * myDests + k), static_cast<int32_t>(off + rows));
                        sent = true;
                    }
                }
            }
            // ── ② F 暂存转发行（行自带 32B 头，直接从暂存环 GM 远程写；仅 owner 槽有）──
            if (dIdx < ranksPerSupernode && fCoreCnt > 0U) {
                for (uint32_t f = 0; f < fCoreCnt; ++f) {
                    uint64_t idx = static_cast<uint64_t>(f) * ranksPerSupernode + dIdx;
                    int32_t staged = Rd32(relayStagedGM_ + idx * 4);
                    int32_t drained = Rd32(relayDrainedGM_ + idx * 4);
                    while (drained < staged) {
                        uint32_t run = static_cast<uint32_t>(staged - drained);
                        uint32_t ringPos = static_cast<uint32_t>(drained) % batchRows_;
                        uint32_t fit = batchRows_ - ringPos; // 源环不跨尾
                        if (run > fit) {
                            run = fit;
                        }
                        if (!sender.HasCredit(run)) {
                            sender.AskCredit(handle, area);
                            break; // 信用不足：先转其它目的/其它 F 核
                        }
                        GM_ADDR src = relayGM_ + idx * batchRows_ * win_.rowStride +
                                      static_cast<uint64_t>(ringPos) * win_.rowStride;
                        sender.SendBatch(handle, area, src, run);
                        sender.DrainChannel(handle); // 通道取走源环行后才许 F 复写——先排空再推进
                        drained += static_cast<int32_t>(run);
                        Wr32(relayDrainedGM_ + idx * 4, drained);
                        sent = true;
                        staged = Rd32(relayStagedGM_ + idx * 4); // 消费中 F 可能续备
                    }
                }
            }
            sflow.SetValue(static_cast<int32_t>(k), static_cast<int32_t>(sender.WriteCnt()));
            sflow.SetValue(static_cast<int32_t>(myDests + k),
                           sender.SyncCredit()); // 收编在途信用读（防跨目的互踩 scratch）
            if (sent) {
                progressed = true;
            }
            // ── ③ 完结判定：自有行发完 + 全部 F 核 EOF + 暂存排空（轨道目的无暂存）──
            bool ownDone = (taskIdx < 0) || (sflow.GetValue(static_cast<int32_t>(3 * myDests + k)) >= taskRows);
            if (ownDone) {
                bool relayDone = true;
                if (dIdx < ranksPerSupernode && fCoreCnt > 0U) {
                    for (uint32_t f = 0; f < fCoreCnt && relayDone; ++f) {
                        uint64_t idx = static_cast<uint64_t>(f) * ranksPerSupernode + dIdx;
                        relayDone = (Rd32(relayEofGM_ + idx * 4) != 0) &&
                                    (Rd32(relayDrainedGM_ + idx * 4) == Rd32(relayStagedGM_ + idx * 4));
                    }
                }
                if (relayDone) {
                    sender.Finish(handle, area); // Drain + done = writeCnt+1（本地单写，无聚合）
                    sflow.SetValue(static_cast<int32_t>(4 * myDests + k), 1);
                    unfinished--;
                    progressed = true;
                }
            }
        }
        if (unfinished == 0U) {
            break;
        }
        if (!progressed) {
            // 整圈无进展：自旋完成在途信用读后重转
            prog(20);
            ++idleSpin;
            ascendc_assert(WinNowUs() - idleT0 < WIN_TIMEOUT_US,
                           "rail send idle timeout: aiv=%u, unfinished=%u, prog=%d", aivId_, unfinished,
                           static_cast<int32_t>(Rd32(segCountGM_ + static_cast<uint64_t>(aivId_) * 4)));
            for (uint32_t k = 0; k < myDests; ++k) {
                if (sflow.GetValue(static_cast<int32_t>(4 * myDests + k)) != 0) {
                    continue;
                }
                uint32_t dIdx = sCoreIdx + k * sCoreCnt;
                uint64_t handle = 0;
                GM_ADDR area = nullptr;
                destAddr(dIdx, handle, area);
                RowSender sender;
                sender.Init(hcomm_, counterScratchGM_ + aivId_ * UB_ALIGN, tpipe_, creditBuf_.Get<int32_t>(), win_,
                            aivId_, myGlobalRank_, static_cast<uint32_t>(sflow.GetValue(static_cast<int32_t>(k))),
                            sflow.GetValue(static_cast<int32_t>(myDests + k)));
                sender.WaitCredit(handle, area);
                sflow.SetValue(static_cast<int32_t>(myDests + k), sender.RemoteReadCnt());
            }
        } else {
            idleSpin = 0;
            idleT0 = WinNowUs(); // 有进展——计时复位
        }
    }
    prog(29); // 全部目的完结
    // DFX：Σ属主流 writeCnt → flagScratch[1]
    int32_t sentSum = 0;
    for (uint32_t k = 0; k < myDests; ++k) {
        sentSum += sflow.GetValue(static_cast<int32_t>(k));
    }
    AscendC::AtomicAdd(reinterpret_cast<__gm__ int32_t*>(flagScratchGM_ + 4), sentSum);
    Wr32(segCountGM_ + aivId_ * 4, 0); // S 核不收集（Σ 交叉校验覆盖全核）
}

// ── F 核：纯消费者 + 暂存（不碰通道）。单相消费 rail 区，按行头 dst 分类：dst==self
//    本地收集；否则暂存 owner 槽（键 L_T）由属主 S 核直投。行原样拷入本核私有环格并
//    滚动发布 staged；差值 ≥ batchRows 即背压。EOF：rail 区收齐即全部置位（链无环）。
/// @brief F 核纯消费循环：单相消费 rail 区
/// @return 无（aivId_ 不在 F 组时直接返回）
__aicore__ inline void EngramFetchGradRailAlignedArch35::FwdLoopF()
{
    if (aivId_ < sendCoreEnd_ || aivId_ >= fwdCoreEnd_) {
        return;
    }
    uint32_t fCoreIdx = aivId_ - sendCoreEnd_;
    uint32_t fCoreCnt = fwdCoreEnd_ - sendCoreEnd_;
    if (fCoreCnt == 0U) {
        return;
    }
    const uint32_t ranksPerSupernode = topo_.ranksPerSupernode;
    const uint32_t slots = ranksPerSupernode; // [0,ranksPerSupernode)=owner 槽（键 L_T）
    uint32_t cursor = 0; // 收集行数（Σ 交叉校验用；落位由协议 pos 决定，不按核分段）

    // 本核暂存状态：每槽滚动计数（UB 私有视图；GM 侧 relayStaged 同步滚动发布）
    LocalTensor<int32_t> staged = flowBuf_.Get<int32_t>();
    for (uint32_t i = 0; i < slots; ++i) {
        staged.SetValue(static_cast<int32_t>(i), 0);
    }
    auto relaySlot = [&](uint32_t slot) __attribute__((cce_aicore))
    {
        return relayGM_ + (static_cast<uint64_t>(fCoreIdx) * slots + slot) * batchRows_ * win_.rowStride;
    };
    auto stageRow = [&](GM_ADDR rowGM, uint32_t slot) __attribute__((cce_aicore))
    {
        uint64_t idx = static_cast<uint64_t>(fCoreIdx) * slots + slot;
        int32_t n = staged.GetValue(static_cast<int32_t>(slot));
        // 背压：环满（staged − drained ≥ batchRows）等属主 S 核排空再复写该行格（真墙钟 60s）
        uint64_t t0 = WinNowUs();
        while (n - Rd32(relayDrainedGM_ + idx * 4) >= static_cast<int32_t>(batchRows_)) {
            ascendc_assert(WinNowUs() - t0 < WIN_TIMEOUT_US,
                           "rail relay backpressure timeout: f=%u, slot=%u, staged=%d, drained=%d", fCoreIdx, slot, n,
                           static_cast<int32_t>(Rd32(relayDrainedGM_ + idx * 4)));
        }
        WinLocalCopy(relaySlot(slot) + static_cast<uint64_t>(n % batchRows_) * win_.rowStride, rowGM, win_.rowStride,
                     tpipe_, entryBuf_.Get<uint8_t>(), gradBuf_.Get<uint8_t>(), evCopy2To3_, evCopy3To2_);
        staged.SetValue(static_cast<int32_t>(slot), n + 1);
        Wr32(relayStagedGM_ + idx * 4, n + 1); // 行数据（MTE3）完成后才发布计数——顺序保证
    };
    // 统一分类：一行进槽（行自带 32B 头原样暂存；攒批/信用/远程写全部由属主 S 核接管）
    auto fwdRow = [&](GM_ADDR rowGM) __attribute__((cce_aicore))
    {
        uint32_t dst = static_cast<uint32_t>(Rd32(rowGM));
        if (dst == myGlobalRank_) { // 轨道对端即 owner（dst==self）→ 直接本地收集
            int32_t src = Rd32(rowGM + 4);
            CollectOneRow(rowGM + ROW_HEAD_BYTES, src, cursor); // 收件区行：+32 跳行头
            return;
        }
        uint32_t ownerLocal = 0;
        (void)FwdRoute(topo_, me_, dst, ownerLocal); // 轨道区行 dst 必在本超节点
        stageRow(rowGM, ownerLocal);                 // 暂存 owner 槽（键 = 其本地号 L_T）
    };
    auto publishEof = [&](uint32_t fromSlot, uint32_t toSlot) __attribute__((cce_aicore))
    {
        for (uint32_t s = fromSlot; s < toSlot; ++s) {
            Wr32(relayEofGM_ + (static_cast<uint64_t>(fCoreIdx) * slots + s) * 4, 1);
        }
    };

    // ── 单相：rail-区（写者 = 各超节点同 L 卡的 S 轨道流）──
    Wr32(segCountGM_ + static_cast<uint64_t>(aivId_) * 4, 60); // DFX: F rail 相开始
    for (uint32_t sn = fCoreIdx; sn < topo_.supernodeCnt; sn += fCoreCnt) {
        if (sn == me_.supernode) {
            continue; // 本超节点一格空置（无写者）
        }
        RecvLoop(
            RailArea(localWin_, win_, sn), win_, *tpipe_, creditBuf_.Get<int32_t>(),
            [&](GM_ADDR blockGM, uint32_t rows) __attribute__((cce_aicore)) {
                for (uint32_t j = 0; j < rows; ++j) {
                    fwdRow(blockGM + j * win_.rowStride);
                }
            },
            segCountGM_);
    }
    publishEof(0, ranksPerSupernode);                          // owner 槽仅本相供行——rail 区收齐即 EOF
    Wr32(segCountGM_ + static_cast<uint64_t>(aivId_) * 4, 61); // DFX: F rail 相完成
    // DFX：ΣF 暂存发布数 → flagScratch[2]
    int32_t stagedSum = 0;
    for (uint32_t s = 0; s < slots; ++s) {
        stagedSum += staged.GetValue(static_cast<int32_t>(s));
    }
    AscendC::AtomicAdd(reinterpret_cast<__gm__ int32_t*>(flagScratchGM_ + 8), stagedSum);
    Wr32(segCountGM_ + aivId_ * 4, static_cast<int32_t>(cursor)); // F 核收集游标上报（终值）
}

// O/F 收集单行：rowDataGM 指向行数据本体（收件区行需 +ROW_HEAD_BYTES 跳头；本地桶行无头）。
// 落位 pos = recvPrefix[src] + 该源到达序——协议唯一确定，不按核分段（行分布不均，
// 均摊分段会越界互踩）；pos 单写者由同源落位不变量保证，无竞争。
__aicore__ inline void EngramFetchGradRailAlignedArch35::CollectOneRow(GM_ADDR rowDataGM, int32_t srcRank,
                                                                       uint32_t& cursor)
{
    int64_t arrive = Rd32(srcArriveGM_ + static_cast<uint64_t>(srcRank) * 4);
    Wr32(srcArriveGM_ + static_cast<uint64_t>(srcRank) * 4, static_cast<int32_t>(arrive + 1));
    int64_t pos = Rd32(recvPrefixGM_ + static_cast<uint64_t>(srcRank) * 4) + arrive;
    // 每源到达数不得超记账（超发 = 上游路由/对账错）——fail-fast，防错位滑入邻源段静默污染
    int32_t cnt = Rd32(recvCountsGM_ + static_cast<uint64_t>(srcRank) * 4);
    ascendc_assert(arrive < cnt, "rail per-src overflow: core=%u src=%d arrive=%lld cnt=%d", aivId_, srcRank, arrive,
                   cnt);
    int32_t entry = Rd32(recvLocalEntryGM_ + static_cast<uint64_t>(pos) * 4);
    Wr32(collectEntryGM_ + static_cast<uint64_t>(pos) * 4, entry);
    WinLocalCopy(collectRowsGM_ + static_cast<uint64_t>(pos) * hiddenBytes_, rowDataGM,
                 static_cast<uint64_t>(hiddenBytes_), tpipe_, entryBuf_.Get<uint8_t>(), gradBuf_.Get<uint8_t>(),
                 evCopy2To3_, evCopy3To2_);
    cursor++;
}

// O 收集组：名下 owner 收件区（跳自槽）轮转 + 本地桶任务 → 收集数组；游标上报核 0 汇总
/// @brief O 核收集循环：轮询名下 owner 区 + 本地桶任务，行按协议位落收集数组
/// @return 无（aivId_ 不在 O 组时直接返回）
__aicore__ inline void EngramFetchGradRailAlignedArch35::OwnerLoopO()
{
    if (aivId_ < fwdCoreEnd_) {
        return;
    }
    uint32_t oCoreIdx = aivId_ - fwdCoreEnd_;
    uint32_t oCoreCnt = totalBlocks_ - fwdCoreEnd_;
    if (oCoreCnt == 0U) {
        return;
    }
    uint32_t cursor = 0; // 收集行数（Σ 交叉校验用；落位由协议 pos 决定，不按核分段）
    Wr32(segCountGM_ + static_cast<uint64_t>(aivId_) * 4, 70); // DFX: O 轮询开始
    // 名下 owner 收件区轮转（L' = 写者本地号；ranksPerSupernode 格，自槽 myL 空置）
    for (uint32_t peerLocal = oCoreIdx; peerLocal < topo_.ranksPerSupernode; peerLocal += oCoreCnt) {
        if (peerLocal == me_.local) {
            continue;
        }
        RecvLoop(
            OwnArea(localWin_, win_, peerLocal), win_, *tpipe_, creditBuf_.Get<int32_t>(),
            [&](GM_ADDR blockGM, uint32_t rows) __attribute__((cce_aicore)) {
                for (uint32_t j = 0; j < rows; ++j) {
                    GM_ADDR rowGM = blockGM + j * win_.rowStride;
                    int32_t src = Rd32(rowGM + 4);
                    CollectOneRow(rowGM + ROW_HEAD_BYTES, src, cursor); // 收件区行：+32 跳行头
                }
            },
            segCountGM_);
    }
    // 本地桶（dst==myRank，src=myRank；行取自 gradSorted，无行头）
    int32_t taskCnt = Rd32(segCountsGM_);
    for (int32_t taskIdx = static_cast<int32_t>(oCoreIdx); taskIdx < taskCnt;
         taskIdx += static_cast<int32_t>(oCoreCnt)) {
        GM_ADDR taskG = taskTableGM_ + static_cast<uint64_t>(taskIdx) * 16; // Rd32 bypass（跨核可见性）
        if (Rd32(taskG) != BUCKET_LOCAL) {
            continue;
        }
        int64_t startRow = Rd32(taskG + 8);
        int64_t cnt = Rd32(taskG + 12);
        for (int64_t r = 0; r < cnt; ++r) {
            // 本地桶行取自 gradSorted——裸数据无行头，直传数据指针（收件区行才 +ROW_HEAD_BYTES，
            // 两类行不可混用，混用会整体错位 32B）
            CollectOneRow(gradSortedGM_ + static_cast<uint64_t>(startRow + r) * hiddenBytes_,
                          static_cast<int32_t>(myGlobalRank_), cursor);
        }
    }
    Wr32(segCountGM_ + aivId_ * 4, static_cast<int32_t>(cursor));
}

// ═══════════════ 阶段 C ═══════════════

__aicore__ inline void EngramFetchGradRailAlignedArch35::SoftBarrier()
{
    // 排序软屏障（分轨排序全核参与）：AtomicAdd 到达 + bypass 自旋等待
    if (totalBlocks_ <= 1U) {
        softBarrierGen_++;
        return;
    }
    ascendc_assert(softBarrierGen_ < SOFT_BARRIER_MAX_GENS, "rail soft barrier overflow: gen=%u", softBarrierGen_);
    __gm__ int32_t* counterPtr = reinterpret_cast<__gm__ int32_t*>(
        sortBarrierGM_ + static_cast<uint64_t>(softBarrierGen_) * softBarrierPageInts_ * sizeof(int32_t));
    // PIPE_ALL：AtomicAdd 到达前须保证本核在飞流水完成（对端越过屏障即读本核产物），
    // SortLib 内部流水混合不可逐一事件收尾，取全流水排空
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::AtomicAdd(counterPtr, 1);
    while (AscendC::ReadGmBypassDCache(counterPtr) < static_cast<int32_t>(totalBlocks_)) {
        for (volatile int32_t d = 0; d < static_cast<int32_t>(SPIN_PAUSE_ITERS); ++d) {
        }
    }
    softBarrierGen_++;
}

__aicore__ inline void EngramFetchGradRailAlignedArch35::RunSortPhase(uint32_t collected)
{
    tpipe_->ReleaseEventID<AscendC::HardEvent::MTE2_MTE3>(evMte2ToMte3_[0]);
    tpipe_->ReleaseEventID<AscendC::HardEvent::MTE2_MTE3>(evMte2ToMte3_[1]);
    tpipe_->ReleaseEventID<AscendC::HardEvent::MTE3_MTE2>(evMte3ToMte2_[0]);
    tpipe_->ReleaseEventID<AscendC::HardEvent::MTE3_MTE2>(evMte3ToMte2_[1]);
    tpipe_->ReleaseEventID<AscendC::HardEvent::MTE2_MTE3>(evCopy2To3_[0]);
    tpipe_->ReleaseEventID<AscendC::HardEvent::MTE2_MTE3>(evCopy2To3_[1]);
    tpipe_->ReleaseEventID<AscendC::HardEvent::MTE3_MTE2>(evCopy3To2_[0]);
    tpipe_->ReleaseEventID<AscendC::HardEvent::MTE3_MTE2>(evCopy3To2_[1]);
    tpipe_->Reset();
    softBarrierGen_ = 0;
    SortLib::SortParams sortParams;
    sortParams.numTileData = sortNumTileData_;
    sortParams.tileCount = (collected + sortNumTileData_ - 1U) / sortNumTileData_;
    sortParams.activeCores = (sortParams.tileCount < totalBlocks_) ? sortParams.tileCount : totalBlocks_;
    sortParams.tmpUbSize = sortTmpUbSize_;
    sortParams.totalElements = static_cast<int64_t>(collected);
    sortParams.isSingleCore = 0;
    sortParams.jobIndex = aivId_;
    sortParams.totalJobs = totalBlocks_;
    // 值域须覆盖全局上界 N×world（uniqueScatter 输出侧减 rankBase 转本地）：
    // 按本地 N 裁剪会丢高位差异位 → 同段两值错序 → 去重漏并
    uint64_t maxKeyId = static_cast<uint64_t>(numEntriesPerRank_) * static_cast<uint64_t>(topo_.world);
    maxKeyId = (maxKeyId > 0ULL) ? maxKeyId - 1ULL : 0ULL;
    uint32_t keyBits = 0;
    while (maxKeyId != 0ULL) {
        keyBits++;
        maxKeyId >>= 1;
    }
    constexpr uint32_t kRadixBits = 8U;
    uint32_t radixRounds = (keyBits + kRadixBits - 1U) / kRadixBits;
    constexpr uint32_t kTrimMinCores = 12U;
    if (radixRounds == 0U || sortParams.activeCores < kTrimMinCores) {
        radixRounds = 0U; // 0 = SortLib 回退全 sizeof(ValT) 轮
    }
    sortParams.radixRounds = radixRounds;
    EngramRailSortSoftBarrier softBarrier{this};
    SortLib::SortInvoke<int32_t, int32_t, uint32_t, false, EngramRailSortSoftBarrier>(
        tpipe_, reinterpret_cast<__gm__ int32_t*>(collectEntryGM_), reinterpret_cast<__gm__ int32_t*>(collectEntryGM_),
        reinterpret_cast<__gm__ int32_t*>(sortCompanionGM_), reinterpret_cast<__gm__ char*>(sortWorkspaceGm_),
        sortParams, softBarrier);
}

// ═══════════════ 主流程 ═══════════════

/// @brief 主流程：barrier 两轮清区 -> A（unsort/桶表/前缀）-> B（S/F/O 并发）-> C（排序去重）
/// @return 无（非 AIV 核直接返回）
__aicore__ inline void EngramFetchGradRailAlignedArch35::Process()
{
    if (g_coreType != AscendC::AIV) { // ASCEND_IS_AIV 宏展开为 constexpr(...)，不支持 ! 取反——g_coreType 直接比较
        return;
    }
    uint32_t numRecv = ReadNumRecv();
    InitCoreGroups();

    // ── barrier 轮① + 清区 + 轮②（两轮协议；epoch 槽跨调用单调推进，槽免清零）──
    uint32_t epoch = static_cast<uint32_t>(
        Rd32(localWin_ + static_cast<uint64_t>(topo_.writerSlots) * STATE_OFFSET)); // = done(上轮) = 2c
    BarrierWaitWritersDone(epoch); // 等全部写者上轮收尾（首次调用 epoch=0 直通）
    ClearRecvAreas();              // W 收件区头 + srcArrive + relay 三表清零（分片）
    AscendC::SyncAll<true>();
    BarrierAnnounceCleared(epoch + 1U); // 通告"已清可写"（clr(c)=2c+1），写者收到才许写我

    // ── 阶段 A：全核 ──
    UnsortGradRail();
    BuildSendPrefix();
    BuildRecvPrefix();
    if (numRecv > 0) {
        // soft barrier 区显式清零（防首调 workspace 垃圾）
        if (aivId_ == 0) {
            LocalTensor<int32_t> zeroLocal = tempBuf_.Get<int32_t>();
            AscendC::Duplicate<int32_t>(zeroLocal, 0, softBarrierPageInts_);
            SyncEv<AscendC::HardEvent::V_MTE3>();
            GlobalTensor<int32_t> softBarrierGM;
            softBarrierGM.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(sortBarrierGM_));
            uint64_t pageInts = softBarrierPageInts_;
            AscendC::DataCopyParams pageCp{1U, static_cast<uint16_t>(softBarrierPageInts_ * 4U), 0U, 0U};
            for (uint32_t g = 0; g < SOFT_BARRIER_MAX_GENS; ++g) {
                AscendC::DataCopyPad(softBarrierGM[g * pageInts], zeroLocal, pageCp);
            }
            SyncEv<AscendC::HardEvent::MTE3_S>();
        }
    }
    MarkOwnerSorted();
    BucketClassify();
    AscendC::SyncAll<true>();
    MergeSegmentsBySGroup(); // S0 合并 → 任务表
    AscendC::SyncAll<true>();
    BarrierWaitPeersCleared(epoch + 1U); // ranksPerSupernode 发送门：写者们本轮"已清可写"（clr(c)=2c+1）全部就位

    // ── 阶段 B：三组并发（常驻消费组是死锁硬要求）──
    if (aivId_ < sendCoreEnd_) {
        SendLoopS();
    } else if (aivId_ < fwdCoreEnd_) {
        FwdLoopF();
    } else {
        OwnerLoopO();
    }
    AscendC::SyncAll<true>();

    // 收齐交叉校验：Σcollected == numRecv
    uint32_t collected = numRecv;
    if (aivId_ == 0) {
        uint32_t total = 0;
        for (uint32_t c = 0; c < totalBlocks_; ++c) {
            total += static_cast<uint32_t>(Rd32(segCountGM_ + c * 4));
        }
        // DFX：带任务表口径与发送侧实况，分辨"任务在而无人发"等错因
        int32_t taskCntD = Rd32(segCountsGM_);
        int32_t segSumD = 0;
        for (uint32_t c = 0; c < totalBlocks_; ++c) {
            segSumD += Rd32(segCountsGM_ + (1 + c) * 4);
        }
        int32_t sentRowsD = ReadCnt(flagScratchGM_, 4); // S 属主流 ΣwriteCnt（AtomicAdd）
        ascendc_assert(total == numRecv,
                       "rail collected(%u)!=numRecv(%u): T=%lld segSum=%d taskCnt=%d t0=(k%d,d%d,c%d) "
                       "t1=(k%d,d%d,c%d) sent=%d",
                       total, numRecv, static_cast<int64_t>(numTokens_), segSumD, taskCntD,
                       static_cast<int32_t>(Rd32(taskTableGM_)), static_cast<int32_t>(Rd32(taskTableGM_ + 4)),
                       static_cast<int32_t>(Rd32(taskTableGM_ + 12)), static_cast<int32_t>(Rd32(taskTableGM_ + 16)),
                       static_cast<int32_t>(Rd32(taskTableGM_ + 20)), static_cast<int32_t>(Rd32(taskTableGM_ + 28)),
                       sentRowsD);
        Wr32(flagScratchGM_, static_cast<int32_t>(total));
    }
    AscendC::SyncAll<true>();
    collected = static_cast<uint32_t>(Rd32(flagScratchGM_));

    // ── 调用收尾 barrier：向写者写 done(c)=2c+2（下一调用轮① 的等待目标），核0 推 epoch ──
    BarrierWriteCallDone(epoch + 2U);

    if (collected == 0U) {
        uniqueScatter_.WriteNumUniqueZero(numUniqueOutGM_);
        return;
    }
    // ── 阶段 C：全核（排序 + 去重累加）──
    RunSortPhase(collected);

    tpipe_->Reset(); // C 阶段缓冲重绑（语义同老：sort 用后 Reset 换 unique 布局）
    tpipe_->InitBuffer(statusBuf_, statusBufBytes_);
    tpipe_->InitBuffer(tempBuf_, tempBufBytes_);
    tpipe_->InitBuffer(indicesBuf_, indicesBufBytes_);
    tpipe_->InitBuffer(entryBuf_, uniqueEntryBytes_);
    tpipe_->InitBuffer(gradBuf_, GRAD_BUF_BYTES);
    if (castBufBytes_ > 0U) {
        tpipe_->InitBuffer(castFp32Buf_, castBufBytes_);
    } else {
        tpipe_->InitBuffer(castFp32Buf_, UB_ALIGN);
    }
    tpipe_->InitBuffer(accumBuf_, accumBufBytes_);
    uniqueScatter_.SetCastBuf(castFp32Buf_);
    uniqueScatter_.SetAccumBuf(accumBuf_);
    uniqueScatter_.SetGradSubBatch(gradSubBatch_);
    uniqueScatter_.SetEntryBufBytes(uniqueEntryBytes_);
    uniqueScatter_.CountUniquesParallel(collected, reinterpret_cast<GM_ADDR>(collectEntryGM_), coreStartGM_,
                                        segCountGM_);
    AscendC::SyncAll<true>();
    uniqueScatter_.RunScatterCast(collected, reinterpret_cast<GM_ADDR>(collectEntryGM_), uniqueLocalEntryOutGM_,
                                  numUniqueOutGM_, gradUniqueOutGM_, collectRowsGM_, coreStartGM_, segCountGM_,
                                  sortCompanionGM_);
}

} // namespace Mc2Kernel

#endif // ENGRAM_FETCH_GRAD_RAILALIGNED_ARCH35_H
