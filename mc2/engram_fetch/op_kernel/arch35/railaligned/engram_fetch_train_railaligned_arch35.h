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
 * \file engram_fetch_train_railaligned_arch35.h
 * \brief EngramFetch 分轨（rail-aligned）训练正向 kernel 主类。
 *        五阶段：A 全核（barrier 两轮清区 + 输入排序 + 桶表）→ B1 REQ 流（S/F/O 三组并发）
 *        → C1 全核（recv_counts/rdispls 定稿 + 稳定散射）→ B2 RESP 流（同构换 resp 族）
 *        → C2 全核（Σ 交叉校验 + num_recv + 调用收尾）。
 *        双流复用反向 S/F/O 收发循环（S 统一发送 / F 暂存环 / 非阻塞信用 API，win.h 零修改）。
 *        WIN 四区（两族 × rail|owner）：rail-req | owner-req | rail-resp | owner-resp；
 *        req 族行距 32B，resp 族行距 align32(32+hidden+sf)。
 *        感知只取一数（ranksPerSupernode，attr 透传），超节点数全推导；
 *        全对全计数交换删除（在线记账替代），tokenStaging 中转删除（sStage 直发）。
 */

#ifndef ENGRAM_FETCH_TRAIN_RAILALIGNED_ARCH35_H
#define ENGRAM_FETCH_TRAIN_RAILALIGNED_ARCH35_H

#include "../../engram_fetch_tiling_data.h"
#ifndef ASCENDC_ENGRAM_FETCH_GRAD_TILING_H
#define ASCENDC_ENGRAM_FETCH_GRAD_TILING_H
#endif
#if __has_include("adv_api/hcomm/hcomm.h")
#include "adv_api/hcomm/hcomm.h"
#endif
#include "../../../../engram_fetch_grad/op_kernel/arch35/railaligned/engram_fetch_grad_route.h"
#include "../../../../engram_fetch_grad/op_kernel/arch35/railaligned/engram_fetch_grad_win.h"
#include "../../../../engram_fetch_grad/op_kernel/arch35/sort/sort_lib.h"
#include "adv_api/hccl/hccl.h"

#if defined(ENABLE_ENGRAM_FETCH_KERNEL)

namespace Mc2Kernel {

using namespace AscendC;

// req 族行距恒 32B：REQ 行 = 32B 行头（dst/src/entry/rsvd 骑头部保留区），无行数据
constexpr uint32_t FWD_REQ_ROW_STRIDE = 32U;

class EngramFetchTrainRailAlignedArch35 {
public:
    __aicore__ inline EngramFetchTrainRailAlignedArch35() = default;

    template <bool HasSf>
    __aicore__ inline void Init(GM_ADDR commContext, GM_ADDR indices, GM_ADDR fetched, GM_ADDR fetchedSf,
                                GM_ADDR permOut, GM_ADDR sendCountsOut, GM_ADDR recvCountsOut,
                                GM_ADDR recvLocalEntryOut, GM_ADDR numRecvOut, GM_ADDR workspaceGM,
                                GM_ADDR localStorageAddr, GM_ADDR sfTableAddr, TPipe* pipe,
                                const EngramFetchTilingData* tilingData);

    template <bool HasSf>
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

    __aicore__ inline void Wr32(GM_ADDR p, int32_t v)
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

    __aicore__ inline void LoadTilingData(const EngramFetchTilingData* t);
    __aicore__ inline void CalcWorkSpaceOffset(const EngramFetchTilingData* t);
    __aicore__ inline void InitPipeAndBuffers();
    __aicore__ inline void InitCoreGroups();

    // ── barrier 两轮协议（与反向逐字相同）──
    __aicore__ inline uint32_t PeerOfWriterSlot(uint32_t slot);
    __aicore__ inline bool IsSelfWriterSlot(uint32_t slot);
    __aicore__ inline void BarrierWaitWritersDone(uint32_t epoch);
    __aicore__ inline void BarrierAnnounceCleared(uint32_t epoch);
    __aicore__ inline void BarrierWaitPeersCleared(uint32_t epoch);
    __aicore__ inline void BarrierWriteCallDone(uint32_t doneVal);
    __aicore__ inline void CheckWinHeadInvariant(uint32_t epoch); // 窗头入参一致性自检（违例即 trap）

    // ── 阶段 A ──
    __aicore__ inline void ClearRecvAreas();
    __aicore__ inline void ClearGmRange(GM_ADDR base, int64_t r0, int64_t r1);
    __aicore__ inline void SortInputIndicesRail();
    __aicore__ inline void BucketClassify();
    __aicore__ inline void MergeSegmentsBySGroup(bool accumulateCounts);
    __aicore__ inline void BuildSendPrefix();
    __aicore__ inline void BuildRecvPrefixFinal();

    // ── B1/C1：记账与定稿 ──
    __aicore__ inline void CollectReqRow(int32_t entry, int32_t srcRank);
    __aicore__ inline void WriteRecvCounts();
    __aicore__ inline void StableScatterCollect();
    __aicore__ inline void CrossCheckFillAndClear();
    __aicore__ inline void BuildRespStrips();
    __aicore__ inline void ClearRelayTables();

    // ── B2/C2：响应散位 ──
    template <bool HasSf>
    __aicore__ inline void ScatterRespRow(GM_ADDR rowGM);
    template <bool HasSf>
    __aicore__ inline void ScatterLocalRespRow(int32_t entry, uint32_t arrive);
    __aicore__ inline void SumRespCounts();

    // ── 三组循环（两子阶段翻转：IsReq = B1/req 族，!IsReq = B2/resp 族）──
    template <bool IsReq, bool HasSf>
    __aicore__ inline void SendLoopS();
    template <bool IsReq, bool HasSf>
    __aicore__ inline void FwdLoopF();
    template <bool IsReq, bool HasSf>
    __aicore__ inline void OwnerLoopO();

    // ── 成员 ──
    __gm__ EngramRailCtxHead* ctxHead_{nullptr}; // 平铺 rail ctx 头（localId/supernodeId/supernodeCnt/C；
                                                 // node 表经 NodeWinAt/NodeChannelAt 访问，与推理 node table 同构）
    RankCoord me_{};
    RailTopo topo_{}; // ranksPerSupernode 感知（attr）+ 超节点数/轨道数推导（kernel 唯一权威）
    uint32_t myGlobalRank_{0};
    WinLayout winReq_{};         // req 族布局：rowStride=32、四区中 req 侧两段基址
    WinLayout winResp_{};        // resp 族布局：rowStride=P_resp、四区中 resp 侧两段基址
    uint64_t reqFamilyBytes_{0}; // req 族总字节（四区中 resp 族基址的偏移前提）
    GM_ADDR localWin_{nullptr};  // = NodeWinAt(ctx, localId)（本卡 WIN 基址；node 表自槽）
    uint32_t aivId_{0};
    uint32_t totalBlocks_{1};
    int32_t numEntriesPerRank_{0};
    int64_t numTokens_{0};
    int64_t hiddenBytes_{0};
    int64_t sfBytes_{0};
    uint64_t sfOffBytes_{0}; // resp 行内 sf 起始偏移（数据区起点 + align32(hiddenBytes)，UB 侧恒 32B 对齐）
    uint64_t cap_{0};        // 收集数组容量 = recv_local_entry.dim0 = tilingData.totalRecv
    uint32_t batchRowsReq_{1};
    uint32_t batchRowsResp_{1};
    // sStage/F 暂存环单格步距 = max(两族批块字节)：只按一族开格会把另一族批尾行
    // 溢出进相邻格，两核互踩行数据 → 异主条目错投
    int64_t stageRingStride_{0};
    uint32_t sortNumTileData_{0};
    uint32_t sortTmpUbSize_{0};
    uint32_t collectCursor_{0}; // 本核收集数组游标（B1 在线记账；C1 稳定散射扫描本段）

    // 输入/输出（GM_ADDR 展开含 *，只能单声明符一行——多声明符会把后续字段退化成 __gm__ uint8_t）
    GM_ADDR indicesGM_{nullptr};
    GM_ADDR fetchedGM_{nullptr};
    GM_ADDR fetchedSfGM_{nullptr};
    GM_ADDR permOutGM_{nullptr};
    GM_ADDR sendCountsOutGM_{nullptr};
    GM_ADDR recvCountsOutGM_{nullptr};
    GM_ADDR recvLocalEntryOutGM_{nullptr};
    GM_ADDR numRecvOutGM_{nullptr};
    GM_ADDR workspaceGM_{nullptr};
    uint64_t localStorageAddr_{0};
    __gm__ uint8_t* sfTableGM_{nullptr};

    // workspace 段（与 engram_fetch_tiling.cpp 分轨段 SetRailWorkSpace 逐段一致）
    GM_ADDR sortedIndicesGM_{nullptr};
    GM_ADDR srcSortedGM_{nullptr}; // C1 伴随：recv_local_entry 每行位的来源 src（B2 S 逐行取行头 dst）
    GM_ADDR bucketSegGM_{nullptr};
    GM_ADDR relayGM_{nullptr}; // F 核暂存环：[fCoreCnt][ranksPerSupernode 槽][batchRowsResp×P_resp]——F 私有格
    GM_ADDR relayStagedGM_{nullptr}; // [fCoreCnt][ranksPerSupernode]：F 已备行数（滚动累计，F 单写）
    GM_ADDR relayDrainedGM_{nullptr}; // [fCoreCnt][ranksPerSupernode]：属主 S 核已发行数（滚动游标，S 单写）
    GM_ADDR relayEofGM_{nullptr}; // [fCoreCnt][ranksPerSupernode]：EOF 标记（F 相位完结后写 1）
    GM_ADDR sStageGM_{nullptr};
    GM_ADDR collectGM_{nullptr};   // [aivNum][cap]×(src,entry) 对——每核独立游标段（无竞争）
    GM_ADDR srcArriveGM_{nullptr}; // world×4B：B1 每源到达计数；C1 改作 fill[]；B2 改作每源响应计数
    GM_ADDR recvPrefixGM_{nullptr}; // world+1：rdispls（int32）+ [world]=Σ recv_counts
    GM_ADDR sendPrefixGM_{nullptr}; // world+1：sdispls（int32）+ [world]=T
    GM_ADDR counterScratchGM_{nullptr};
    GM_ADDR flagScratchGM_{nullptr};
    GM_ADDR segCountGM_{nullptr};  // [0]=taskCount, [1+c]=核 c 段数；C2 兼作 Σ 分片暂存
    GM_ADDR coreStartGM_{nullptr}; // 前缀两遍 scratch
    GM_ADDR dfxGM_{nullptr};       // 每核进度码（S 10/20/29；F 60/61/62/63；O 70/71）
    GM_ADDR sortWorkspaceGm_{nullptr};
    // bucketSeg 内部布局
    GM_ADDR segCountsGM_{nullptr}; // 每核段数头（含 taskCount 在最前）
    GM_ADDR stripsGM_{nullptr};    // 核条带 (dst,start,cnt)×3 int32
    GM_ADDR taskTableGM_{nullptr}; // 任务表 (kind,dst,start,cnt)×4 int32
    uint32_t stripCap_{0};

    // 三组边界
    uint32_t sendCoreEnd_{0};
    uint32_t fwdCoreEnd_{0};

    // UB
    AscendC::TPipe* tpipe_{nullptr};
    const EngramFetchTilingData* tiling_{nullptr}; // 排序后重建常驻缓冲时复用（老 train 同款时序）
    TBuf<> hcommBuf_;
    TBuf<> creditBuf_;
    TBuf<> rowBlockBuf_;
    TBuf<> indicesBuf_;
    TBuf<> statusBuf_;
    TBuf<> tempBuf_;
    TBuf<> entryBuf_;
    TBuf<> gradBuf_;
    TBuf<> flowBuf_;
    TBuf<> sFlowBuf_;
    uint32_t indicesBufElems_{0};
    uint32_t flowBufBytes_{0};
    AscendC::Hcomm<COMM_PROTOCOL_UBC_CTP> hcomm_;
    int32_t evCopy2To3_[2] = {0, 0}; // WinLocalCopy 乒乓（行搬运）
    int32_t evCopy3To2_[2] = {0, 0};
    // 超时统一走 win.h 的真墙钟 WIN_TIMEOUT_US（60s）
};

/// @brief tilingData 装载与运行期派生：拓扑、双族 WIN 布局、暂存环步距
__aicore__ inline void EngramFetchTrainRailAlignedArch35::LoadTilingData(const EngramFetchTilingData* t)
{
    numEntriesPerRank_ = t->numEntriesPerRank;
    numTokens_ = t->numTokens;
    hiddenBytes_ = t->hiddenBytes;
    sfBytes_ = static_cast<uint64_t>(t->numSfPacks) * static_cast<uint64_t>(t->sfElemSize);
    // resp 行内布局：[32B 头][hidden 数据][pad→32B][sf][pad→P_resp]
    // —— sfOff 对齐保证 S 拼块时全部 UB 侧 MTE 目的地址 32B 对齐
    sfOffBytes_ = (hiddenBytes_ + UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN;
    cap_ = t->totalRecv;
    batchRowsReq_ = (t->batchRowsReq > 0U) ? t->batchRowsReq : 1U;
    batchRowsResp_ = (t->batchRowsResp > 0U) ? t->batchRowsResp : 1U;
    sortNumTileData_ = t->sortNumTileData;
    sortTmpUbSize_ = t->sortTmpUbSize;
    topo_ = MakeRailTopo(t->ranksPerSupernode, t->rankSize);
    // ═══ 双族布局（两族共用一个 N；req 族在前、resp 族在后；每族 rail|owner 两区）═══
    const uint32_t num = (t->numRows > 0U) ? t->numRows : 1U;
    const uint32_t snCnt = topo_.supernodeCnt;
    winReq_.topo = topo_;
    winReq_.rowStride = FWD_REQ_ROW_STRIDE;
    winReq_.numRows = num;
    winReq_.areaSize = RECV_AREA_HEAD + static_cast<uint64_t>(num) * winReq_.rowStride;
    winReq_.winHead = static_cast<uint64_t>(topo_.writerSlots + EPOCH_SLOT_TAIL) * STATE_OFFSET;
    winReq_.railBase = winReq_.winHead;
    winReq_.ownBase = winReq_.railBase + static_cast<uint64_t>(snCnt) * winReq_.areaSize;
    reqFamilyBytes_ = static_cast<uint64_t>(snCnt + topo_.ranksPerSupernode) * winReq_.areaSize;
    winResp_.topo = topo_;
    winResp_.rowStride =
        static_cast<uint32_t>((ROW_HEAD_BYTES + sfOffBytes_ + sfBytes_ + UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN);
    winResp_.numRows = num;
    winResp_.areaSize = RECV_AREA_HEAD + static_cast<uint64_t>(num) * winResp_.rowStride;
    winResp_.winHead = winReq_.winHead;
    winResp_.railBase = winReq_.ownBase + static_cast<uint64_t>(topo_.ranksPerSupernode) * winReq_.areaSize;
    winResp_.ownBase = winResp_.railBase + static_cast<uint64_t>(snCnt) * winResp_.areaSize;

    {
        constexpr uint32_t kBatchBlockBytes = 65504U; // 批块字节预算（DataCopyExtParams.blockLen 为 uint16）
        const uint32_t maxReqRows = kBatchBlockBytes / FWD_REQ_ROW_STRIDE;
        if (batchRowsReq_ > maxReqRows) {
            batchRowsReq_ = maxReqRows;
        }
        if (winResp_.rowStride > 0U) {
            const uint32_t maxRespRows = kBatchBlockBytes / winResp_.rowStride;
            if (batchRowsResp_ > maxRespRows) {
                batchRowsResp_ = maxRespRows;
            }
        }
        // 纵深防御：与 tiling floor 1 同口径——行距错配把 maxRows 钳成 0 时兜回 1，防
        // %batchRows 除零与 rows=min(...,batchRows)=0 的零行批忙转（tiling 已硬拒绝 P>65504）
        if (batchRowsReq_ == 0U) {
            batchRowsReq_ = 1U;
        }
        if (batchRowsResp_ == 0U) {
            batchRowsResp_ = 1U;
        }
        ascendc_assert(batchRowsReq_ > 0U && batchRowsResp_ > 0U, "rail fwd batchRows must be positive");
    }
    stageRingStride_ = static_cast<int64_t>(batchRowsReq_) * FWD_REQ_ROW_STRIDE;
    const int64_t respBlock = static_cast<int64_t>(batchRowsResp_) * winResp_.rowStride;
    if (respBlock > stageRingStride_) {
        stageRingStride_ = respBlock;
    }
}

/// @brief workspace 各段基址计算（与 SetRailWorkSpace 逐段一致）
__aicore__ inline void EngramFetchTrainRailAlignedArch35::CalcWorkSpaceOffset(const EngramFetchTilingData* t)
{
    const int64_t T = numTokens_;
    const uint32_t aiv = (t->aivNum > 0U) ? static_cast<uint32_t>(t->aivNum) : 1U;
    const int64_t world = static_cast<int64_t>(t->rankSize);
    const int64_t cap = cap_;
    const uint32_t ranksPerSupernode = topo_.ranksPerSupernode;
    const int64_t ringStride = stageRingStride_; // 两族批块取大（见 LoadTilingData 注释）
    auto align32 = [](int64_t x) __attribute__((cce_aicore))
    {
        return (x + UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN;
    };

    int64_t off = 0;
    sortedIndicesGM_ = workspaceGM_ + off;
    off += align32(T * 4);
    srcSortedGM_ = workspaceGM_ + off;
    off += align32(cap * 4);
    int64_t stripCap = (T > world ? T : world) / aiv + 2; // 每核段数上界：行/核 + world 切片 + 边界分裂
    stripCap_ = static_cast<uint32_t>(stripCap);
    int64_t bucketSegBytes = align32((1 + aiv) * 4) + static_cast<int64_t>(aiv) * stripCap * 12 + (T + aiv) * 16;
    bucketSegGM_ = workspaceGM_ + off;
    off += align32(bucketSegBytes);
    segCountsGM_ = bucketSegGM_; // [0]=taskCount, [1+c]=核 c 段数
    stripsGM_ = bucketSegGM_ + align32((1 + aiv) * 4);
    taskTableGM_ = stripsGM_ + static_cast<int64_t>(aiv) * stripCap * 12;
    // ═══ F 暂存环 + 三表（两族分时复用：环格固定按 resp 族批容量开，req 族在其内用小步距）═══
    const uint32_t relaySlots = ranksPerSupernode;
    const uint32_t fCoreCnt = (t->fwdCoreCnt > 0U) ? t->fwdCoreCnt : 1U;
    relayGM_ = workspaceGM_ + off;
    off += static_cast<int64_t>(fCoreCnt) * relaySlots * ringStride;
    const int64_t relayCtlInts = static_cast<int64_t>(fCoreCnt) * relaySlots;
    relayStagedGM_ = workspaceGM_ + off;
    off += align32(relayCtlInts * 4);
    relayDrainedGM_ = workspaceGM_ + off;
    off += align32(relayCtlInts * 4);
    relayEofGM_ = workspaceGM_ + off;
    off += align32(relayCtlInts * 4);
    sStageGM_ = workspaceGM_ + off;
    off += static_cast<int64_t>(t->sendCoreCnt) * ringStride;
    collectGM_ = workspaceGM_ + off;
    off += align32(static_cast<int64_t>(aiv) * cap * 8);
    srcArriveGM_ = workspaceGM_ + off;
    off += align32(world * 4);
    recvPrefixGM_ = workspaceGM_ + off;
    off += align32((world + 1) * 4);
    sendPrefixGM_ = workspaceGM_ + off;
    off += align32((world + 1) * 4);
    counterScratchGM_ = workspaceGM_ + off;
    off += static_cast<int64_t>(aiv) * UB_ALIGN;
    // flagScratch：[0..DFX_SCALAR_SLOTS) DFX 汇总槽；其后每 S 核 DFX_SNAP_WORDS 字失败快照
    flagScratchGM_ = workspaceGM_ + off;
    off += (DFX_SCALAR_SLOTS + static_cast<int64_t>((t->sendCoreCnt > 0U) ? t->sendCoreCnt : 1U) * DFX_SNAP_WORDS) *
           sizeof(int32_t);
    segCountGM_ = workspaceGM_ + off;
    off += align32(aiv * 4);
    coreStartGM_ = workspaceGM_ + off;
    off += align32(aiv * 4);
    dfxGM_ = workspaceGM_ + off;
    off += align32(aiv * 4);

    constexpr int64_t kRadixRounds = 4;
    int64_t slTileCount = static_cast<int64_t>(t->sortTileCount);
    int64_t seg0 = align32(256 * kRadixRounds * 4);
    int64_t seg1 = align32(slTileCount * 256 * kRadixRounds * 4);
    int64_t seg2 = align32(T * 4);
    int64_t seg3 = 2 * align32(slTileCount * 256 * 2);
    int64_t seg4 = align32(slTileCount * t->sortNumTileData);
    int64_t seg5 = align32(T * 4);
    sortWorkspaceGm_ = workspaceGM_ + off;
    off += seg0 + seg1 + seg2 + seg3 + seg4 + seg5;
}

__aicore__ inline void EngramFetchTrainRailAlignedArch35::InitPipeAndBuffers()
{
    const uint32_t tAiv = (tiling_->aivNum > 0U) ? tiling_->aivNum : 1U;
    const uint32_t tSend = (tiling_->sendCoreCnt > 0U) ? tiling_->sendCoreCnt : 1U;
    const uint32_t tFwd = (tiling_->fwdCoreCnt > 0U) ? tiling_->fwdCoreCnt : 1U;
    tpipe_->InitBuffer(hcommBuf_, HCOMM_INIT_SIZE);
    hcomm_.Init(hcommBuf_.Get<uint8_t>(), HCOMM_INIT_SIZE);

    tpipe_->InitBuffer(creditBuf_, UB_ALIGN); // 信用读自旋/标量写暂存
    uint32_t rowBlockBytes =
        static_cast<uint32_t>(stageRingStride_); // 拼行块：两族批块取大
                                                 // （req 满批 65504B > resp 块，按 resp 开会写越界）
    tpipe_->InitBuffer(rowBlockBuf_, rowBlockBytes);
    tpipe_->InitBuffer(indicesBuf_, IDX_BUF_BYTES); // 排序产物分片扫描
    indicesBufElems_ = IDX_BUF_BYTES / sizeof(int32_t);

    uint32_t statusBytes = static_cast<uint32_t>((2 * STATE_OFFSET + static_cast<uint64_t>(tAiv) * 8 + UB_ALIGN - 1) /
                                                 UB_ALIGN * UB_ALIGN);
    tpipe_->InitBuffer(statusBuf_, statusBytes);
    uint32_t coreArrayBytes =
        static_cast<uint32_t>((((static_cast<uint64_t>(tAiv) * 4 + UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN) +
                               static_cast<uint64_t>(tAiv) * 4 + UB_ALIGN - 1) /
                              UB_ALIGN * UB_ALIGN);
    uint32_t tempBytes = (statusBytes > ENTRY_BATCH_CAP * 4U) ? statusBytes : ENTRY_BATCH_CAP * 4U;
    if (tempBytes < coreArrayBytes) {
        tempBytes = coreArrayBytes;
    }
    tpipe_->InitBuffer(tempBuf_, tempBytes);        // 清零 Duplicate 源 + 前缀 scratch
    tpipe_->InitBuffer(entryBuf_, ENTRY_BUF_BYTES); // WinLocalCopy ping
    tpipe_->InitBuffer(gradBuf_, GRAD_BUF_BYTES);   // WinLocalCopy pong

    // F 组流状态：F 核暂存计数（每槽滚动 staged）
    flowBufBytes_ = static_cast<uint32_t>(
        (static_cast<uint64_t>(topo_.ranksPerSupernode) * (sizeof(int32_t)) + UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN);
    tpipe_->InitBuffer(flowBuf_, flowBufBytes_);
    // S 核每目的流状态：5 × 名下目的数 int32（flowW/flowR/taskIdx/ownOff/fin）
    {
        uint32_t sDests = topo_.ranksPerSupernode + topo_.supernodeCnt;
        uint32_t sFlowInts = 5U * ((sDests + tSend - 1U) / tSend);
        tpipe_->InitBuffer(sFlowBuf_,
                           (sFlowInts * static_cast<uint32_t>(sizeof(int32_t)) + UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN);
    }

    evCopy2To3_[0] = static_cast<int32_t>(tpipe_->FetchEventID(AscendC::HardEvent::MTE2_MTE3));
    evCopy2To3_[1] = static_cast<int32_t>(tpipe_->FetchEventID(AscendC::HardEvent::MTE2_MTE3));
    evCopy3To2_[0] = static_cast<int32_t>(tpipe_->FetchEventID(AscendC::HardEvent::MTE3_MTE2));
    evCopy3To2_[1] = static_cast<int32_t>(tpipe_->FetchEventID(AscendC::HardEvent::MTE3_MTE2));
}

// S/F/O 三组（tiling 落表值优先，不一致按 1:2:2 比例兜底；分轨硬要求 ≥4 AIV——三组各 ≥1）
__aicore__ inline void EngramFetchTrainRailAlignedArch35::InitCoreGroups()
{
    if (sendCoreEnd_ > 0U) {
        return;
    }
    uint32_t aiv = totalBlocks_;
    // 1:2:2 兜底配比（tiling 落表值非法时）
    uint32_t sCnt = (aiv >= 5U) ? aiv / 5U : 1U;
    uint32_t fCnt = (aiv >= 5U) ? 2U * aiv / 5U : 1U;
    uint32_t oCnt = (aiv > sCnt + fCnt) ? aiv - sCnt - fCnt : 1U;
    if (sCnt + fCnt + oCnt != aiv || sCnt < 1U || fCnt < 1U || oCnt < 1U) {
        sCnt = 1U;
        fCnt = 1U;
        oCnt = (aiv > 2U) ? aiv - 2U : 1U;
    }
    sendCoreEnd_ = sCnt;
    fwdCoreEnd_ = sendCoreEnd_ + fCnt;
}

/// @brief kernel 初始化：解析平铺 rail ctx（与 tiling 双源校验，违例 trap）、装载 tiling、
///        初始化通信与 UB 缓冲
template <bool HasSf>
__aicore__ inline void EngramFetchTrainRailAlignedArch35::Init(
    GM_ADDR commContext, GM_ADDR indices, GM_ADDR fetched, GM_ADDR fetchedSf, GM_ADDR permOut, GM_ADDR sendCountsOut,
    GM_ADDR recvCountsOut, GM_ADDR recvLocalEntryOut, GM_ADDR numRecvOut, GM_ADDR workspaceGM, GM_ADDR localStorageAddr,
    GM_ADDR sfTableAddr, TPipe* pipe, const EngramFetchTilingData* tilingData)
{
    tpipe_ = pipe;
    indicesGM_ = indices;
    fetchedGM_ = fetched;
    fetchedSfGM_ = fetchedSf;
    permOutGM_ = permOut;
    sendCountsOutGM_ = sendCountsOut;
    recvCountsOutGM_ = recvCountsOut;
    recvLocalEntryOutGM_ = recvLocalEntryOut;
    numRecvOutGM_ = numRecvOut;
    workspaceGM_ = workspaceGM;
    aivId_ = AscendC::GetBlockIdx();
    totalBlocks_ = AscendC::GetBlockNum();

    // 平铺 rail ctx 解析（头部标量 + node 表，纯数据——反向 route.h 复用；
    ctxHead_ = reinterpret_cast<__gm__ EngramRailCtxHead*>(commContext);
    topo_ = MakeRailTopo(tilingData->ranksPerSupernode, tilingData->rankSize);
    uint32_t myLocal = ctxHead_->localId;
    uint32_t mySn = ctxHead_->supernodeId;
    uint32_t expectCnt = topo_.supernodeCnt;
    ascendc_assert(ctxHead_->ranksPerSupernode == topo_.ranksPerSupernode && ctxHead_->supernodeCnt == expectCnt &&
                       mySn * topo_.ranksPerSupernode + myLocal < tilingData->rankSize,
                   "rail fwd ctx mismatch: C=%u(expect %u), snCnt=%u(expect %u), world=%u", ctxHead_->ranksPerSupernode,
                   topo_.ranksPerSupernode, ctxHead_->supernodeCnt, expectCnt, tilingData->rankSize);
    myGlobalRank_ = mySn * topo_.ranksPerSupernode + myLocal; // rank = sn×C + L（头部标量拼出）
    me_ = DecodeRank(topo_, myGlobalRank_);
    localWin_ = reinterpret_cast<GM_ADDR>(NodeWinAt(reinterpret_cast<GM_ADDR>(ctxHead_), myLocal)); // node 表自槽

    GlobalTensor<int64_t> localStorageTensor;
    localStorageTensor.SetGlobalBuffer((__gm__ int64_t*)localStorageAddr);
    localStorageAddr_ = static_cast<uint64_t>(localStorageTensor.GetValue(0));
    if constexpr (HasSf) {
        GlobalTensor<int64_t> sfAddrTensor;
        sfAddrTensor.SetGlobalBuffer((__gm__ int64_t*)sfTableAddr);
        sfTableGM_ = reinterpret_cast<__gm__ uint8_t*>(static_cast<uint64_t>(sfAddrTensor.GetValue(0)));
    }

    LoadTilingData(tilingData);
    CalcWorkSpaceOffset(tilingData);
    tiling_ = tilingData;
    InitPipeAndBuffers();

    // 三组边界：tiling 落表值（sendCoreCnt/fwdCoreCnt/ownerCoreCnt 与 aivNum 一致时优先）
    uint32_t tilingS = tilingData->sendCoreCnt;
    uint32_t tilingF = tilingData->fwdCoreCnt;
    uint32_t tilingO = tilingData->ownerCoreCnt;
    if (tilingS >= 1U && tilingF >= 1U && tilingS + tilingF + tilingO == totalBlocks_) {
        sendCoreEnd_ = tilingS;
        fwdCoreEnd_ = tilingS + tilingF;
    } else {
        sendCoreEnd_ = 0; // 触发 InitCoreGroups 兜底
    }
}

__aicore__ inline bool EngramFetchTrainRailAlignedArch35::IsSelfWriterSlot(uint32_t slot)
{
    const uint32_t ranksPerSupernode = topo_.ranksPerSupernode;
    return slot == me_.local || slot == ranksPerSupernode + me_.supernode;
}

__aicore__ inline uint32_t EngramFetchTrainRailAlignedArch35::PeerOfWriterSlot(uint32_t slot)
{
    const uint32_t ranksPerSupernode = topo_.ranksPerSupernode;
    if (slot < ranksPerSupernode) {
        return me_.supernode * ranksPerSupernode + slot; // 同伴
    }
    return (slot - ranksPerSupernode) * ranksPerSupernode + me_.local; // 轨道对端：sn' 超节点的同 L 卡
}

__aicore__ inline void EngramFetchTrainRailAlignedArch35::BarrierWaitWritersDone(uint32_t target)
{
    const uint32_t W = topo_.writerSlots;
    AscendC::SyncAll<true>();
    uint64_t t0 = WinNowUs(); // 真墙钟 60s（自旋计数受每转成本影响，实测远短于名义值）
    for (uint32_t i = aivId_; i < W; i += totalBlocks_) {
        if (IsSelfWriterSlot(i)) {
            continue; // 自槽永不置位
        }
        while (static_cast<uint32_t>(Rd32(localWin_ + static_cast<uint64_t>(i) * STATE_OFFSET)) < target) {
            ascendc_assert(WinNowUs() - t0 < WIN_TIMEOUT_US, "rail fwd barrier round1 timeout: slot=%u, target=%u", i,
                           target);
        }
    }
    AscendC::SyncAll<true>();
}

__aicore__ inline void EngramFetchTrainRailAlignedArch35::BarrierAnnounceCleared(uint32_t target)
{
    // 通告核 = 该写者槽通道的属主 S 核（与数据发送同核，维持每通道单生产者；hcomm 的
    // sqHead 等为普通 GM 标量写，通告与数据分属两核会按陈旧 sqHead 落 WQE 覆盖 SQ 槽）
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
            Wr32(myScratch, static_cast<int32_t>(target));
            int32_t ret = hcomm_.WriteNbi(handle, peerWin + static_cast<uint64_t>(mySlot) * STATE_OFFSET, myScratch,
                                          STATE_OFFSET);
            ascendc_assert(ret == 0, "rail fwd barrier announce WriteNbi failed, ret=%d, peer=%u", ret, peer);
        }
    }
    AscendC::SyncAll<true>();
}

__aicore__ inline void EngramFetchTrainRailAlignedArch35::BarrierWaitPeersCleared(uint32_t target)
{
    const uint32_t W = topo_.writerSlots;
    AscendC::SyncAll<true>();
    uint64_t t0 = WinNowUs(); // 真墙钟 60s
    for (uint32_t i = aivId_; i < W; i += totalBlocks_) {
        if (IsSelfWriterSlot(i)) {
            continue;
        }
        while (static_cast<uint32_t>(Rd32(localWin_ + static_cast<uint64_t>(i) * STATE_OFFSET)) < target) {
            ascendc_assert(WinNowUs() - t0 < WIN_TIMEOUT_US, "rail fwd barrier round2 timeout: slot=%u, target=%u", i,
                           target);
        }
    }
    AscendC::SyncAll<true>();
}

__aicore__ inline void EngramFetchTrainRailAlignedArch35::BarrierWriteCallDone(uint32_t doneVal)
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
            ascendc_assert(ret == 0, "rail fwd barrier done WriteNbi failed, ret=%d, peer=%u", ret, peer);
        }
    }
    AscendC::SyncAll<true>();
    if (aivId_ == 0) { // epoch 槽推进（下一调用轮① 的等待目标 = 本 done 值）
        Wr32(localWin_ + static_cast<uint64_t>(topo_.writerSlots) * STATE_OFFSET, static_cast<int32_t>(doneVal));
    }
    AscendC::SyncAll<true>();
}

__aicore__ inline void EngramFetchTrainRailAlignedArch35::CheckWinHeadInvariant(uint32_t epoch)
{
    if (aivId_ != 0U) {
        return;
    }
    const uint32_t W = topo_.writerSlots;
    uint32_t doneCnt = 0;
    uint32_t clrCnt = 0;
    uint32_t laggardCnt = 0;
    uint32_t badCnt = 0;
    uint32_t firstSlot = 0;
    int32_t firstVal = 0;
    int32_t minVal = 0x7FFFFFFF;
    int32_t maxVal = -0x7FFFFFFF;
    const bool hasPrevRound = epoch >= 2U;
    for (uint32_t i = 0; i < W; ++i) {
        if (IsSelfWriterSlot(i)) {
            continue; // 自槽永不置位
        }
        int32_t v = Rd32(localWin_ + static_cast<uint64_t>(i) * STATE_OFFSET);
        if (v == static_cast<int32_t>(epoch)) {
            doneCnt++;
            continue;
        }
        if (v == static_cast<int32_t>(epoch + 1U)) {
            clrCnt++;
            continue;
        }
        if (hasPrevRound && v == static_cast<int32_t>(epoch - 1U)) {
            laggardCnt++;
            continue;
        }
        if (v < minVal) {
            minVal = v;
        }
        if (v > maxVal) {
            maxVal = v;
        }
        if (badCnt == 0U) {
            firstSlot = i;
            firstVal = v;
        }
        badCnt++;
    }
    ascendc_assert(badCnt == 0U,
                   "rail fwd win-head GARBAGE at entry: epoch=%u, doneCnt=%u, clrCnt=%u, laggardCnt=%u, "
                   "badSlots=%u, first[slot=%u]=%d, badMin=%d, badMax=%d (legal set is {epoch-1, epoch, epoch+1})",
                   epoch, doneCnt, clrCnt, laggardCnt, badCnt, firstSlot, firstVal, minVal, maxVal);
}

// GM 段清零（[r0, r1) 个 int32，本核分片；Duplicate → DataCopy 分块——DataCopyPad 兼容非对齐尾块）
__aicore__ inline void EngramFetchTrainRailAlignedArch35::ClearGmRange(GM_ADDR base, int64_t r0, int64_t r1)
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

__aicore__ inline void EngramFetchTrainRailAlignedArch35::ClearRecvAreas()
{
    // 全核分片清四区（2W 个收件区 96B 头）+ srcArrive + relay 三表
    const uint32_t ranksPerSupernode = topo_.ranksPerSupernode;
    const uint64_t areasPerFamily = static_cast<uint64_t>(topo_.supernodeCnt) + ranksPerSupernode;
    const uint64_t totalAreas = 2ULL * areasPerFamily;
    LocalTensor<int32_t> zero = tempBuf_.Get<int32_t>();
    AscendC::Duplicate<int32_t>(zero, 0, RECV_AREA_HEAD / sizeof(int32_t));
    SyncEv<AscendC::HardEvent::V_MTE3>();
    for (uint64_t i = aivId_; i < totalAreas; i += totalBlocks_) {
        // 每族 rail|owner 两区（rail 格按源超节点号 sn'、owner 格按写者本地号 L）
        const WinLayout& wfam = (i < areasPerFamily) ? winReq_ : winResp_;
        uint64_t j = (i < areasPerFamily) ? i : i - areasPerFamily;
        GM_ADDR area = nullptr;
        if (j < topo_.supernodeCnt) {
            area = RailArea(localWin_, wfam, static_cast<uint32_t>(j));
        } else {
            area = OwnArea(localWin_, wfam, static_cast<uint32_t>(j - topo_.supernodeCnt));
        }
        GlobalTensor<int32_t> headGM;
        headGM.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(area));
        AscendC::DataCopyParams headCp{1U, static_cast<uint16_t>(RECV_AREA_HEAD), 0U, 0U};
        AscendC::DataCopyPad(headGM, zero, headCp);
    }
    // srcArrive 分片清零（world×4B）
    {
        int64_t world = static_cast<int64_t>(topo_.world);
        int64_t per = (world + totalBlocks_ - 1) / totalBlocks_;
        int64_t s0 = static_cast<int64_t>(aivId_) * per;
        int64_t s1 = (s0 + per > world) ? world : s0 + per;
        ClearGmRange(srcArriveGM_, s0, s1);
    }
    // relay 三表分片清零（staged/drained/eof 连续三段，各 align32(fCoreCnt×ranksPerSupernode×4)——
    // 含段间对齐 padding 一并清零，防止未对齐时 eof 尾部残留脏值）
    {
        const uint32_t fCnt = (fwdCoreEnd_ > sendCoreEnd_) ? (fwdCoreEnd_ - sendCoreEnd_) : 1U;
        const int64_t rawInts = static_cast<int64_t>(fCnt) * static_cast<int64_t>(ranksPerSupernode);
        const int64_t alignInts = (rawInts * 4 + UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN / 4;
        const int64_t ctlCnt = 3 * alignInts;
        int64_t per = (ctlCnt + totalBlocks_ - 1) / totalBlocks_;
        int64_t s0 = static_cast<int64_t>(aivId_) * per;
        int64_t s1 = (s0 + per > ctlCnt) ? ctlCnt : s0 + per;
        ClearGmRange(relayStagedGM_, s0, s1);
    }
    if (aivId_ == 0) {
        // DFX 标量槽 + S 核完结判定快照槽——首调 workspace 为垃圾，快照区不清零会被 trap 误读
        for (uint32_t w = 0; w < DFX_SCALAR_SLOTS + sendCoreEnd_ * DFX_SNAP_WORDS; ++w) {
            Wr32(flagScratchGM_ + w * 4, 0);
        }
    }
}

// 输入排序（SortLib 全核参与，默认硬件屏障）：排序前 Reset 让 SortLib 用满 UB，
// 排序后 Reset + 重建常驻通信缓冲
__aicore__ inline void EngramFetchTrainRailAlignedArch35::SortInputIndicesRail()
{
    tpipe_->ReleaseEventID<AscendC::HardEvent::MTE2_MTE3>(evCopy2To3_[0]);
    tpipe_->ReleaseEventID<AscendC::HardEvent::MTE2_MTE3>(evCopy2To3_[1]);
    tpipe_->ReleaseEventID<AscendC::HardEvent::MTE3_MTE2>(evCopy3To2_[0]);
    tpipe_->ReleaseEventID<AscendC::HardEvent::MTE3_MTE2>(evCopy3To2_[1]);
    tpipe_->Reset();

    SortLib::SortParams p;
    p.numTileData = sortNumTileData_;
    p.tileCount = (numTokens_ + sortNumTileData_ - 1U) / sortNumTileData_;
    p.activeCores =
        (p.tileCount < static_cast<int64_t>(totalBlocks_)) ? static_cast<uint32_t>(p.tileCount) : totalBlocks_;
    p.tmpUbSize = sortTmpUbSize_;
    p.totalElements = numTokens_;
    p.isSingleCore = 0;
    p.jobIndex = aivId_; // 全核模式：物理核号即 job（老 train 同款）
    p.totalJobs = totalBlocks_;
    // radix 轮数按 entry 值域裁剪（V1 反向同款）：entry ∈ [0, numEntriesPerRank×world)
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
    if (radixRounds == 0U || p.activeCores < kTrimMinCores) {
        radixRounds = 0U; // 0 = SortLib 回退全 sizeof(ValT) 轮
    }
    p.radixRounds = radixRounds;
    SortLib::SortInvoke<int32_t, int32_t, uint32_t, false>(
        tpipe_, reinterpret_cast<__gm__ int32_t*>(indicesGM_), reinterpret_cast<__gm__ int32_t*>(sortedIndicesGM_),
        reinterpret_cast<__gm__ int32_t*>(permOutGM_), reinterpret_cast<__gm__ char*>(sortWorkspaceGm_), p);

    tpipe_->Reset();
}

// 三桶条带（阶段 A）：每核扫自己切片 → 本核条带 (dst=owner,start,cnt)
// dst 升序不变量由排序保证（entry 升序 ⇒ owner 非降序）；越界条目立即报错（失效点 #8 最早发现点）
__aicore__ inline void EngramFetchTrainRailAlignedArch35::BucketClassify()
{
    int64_t per = (numTokens_ + totalBlocks_ - 1) / totalBlocks_;
    int64_t start = static_cast<int64_t>(aivId_) * per;
    int64_t end = (start + per > numTokens_) ? numTokens_ : start + per;
    if (start >= end) {
        Wr32(segCountsGM_ + (1 + aivId_) * 4, 0); // 空段也落表（merge 读全核段数）
        return;
    }
    ascendc_assert(numEntriesPerRank_ > 0, "rail fwd numEntriesPerRank(%d) must be positive", numEntriesPerRank_);
    GM_ADDR strip = stripsGM_ + static_cast<uint64_t>(aivId_) * stripCap_ * 12;
    GlobalTensor<int32_t> sortedGM;
    sortedGM.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(sortedIndicesGM_));
    LocalTensor<int32_t> chunk = indicesBuf_.Get<int32_t>();
    uint32_t runCnt = 0;
    int64_t pos = start;
    int32_t curOwner = -1;
    int64_t runStart = start;
    int64_t runLen = 0;
    const int32_t maxEntry =
        static_cast<int32_t>(static_cast<int64_t>(numEntriesPerRank_) * static_cast<int64_t>(topo_.world));
    // 条带落表必须 Wr32（UB→MTE3 DataCopy）：裸标量存 GM 只进本核 L1（写回），SyncAll 不刷
    // 数据缓存——合并核 bypass 读只见陈旧零
    auto emitRun = [&]() __attribute__((cce_aicore))
    {
        Wr32(strip + runCnt * 12 + 0, curOwner);
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
        AscendC::DataCopyPad(chunk, sortedGM[static_cast<uint32_t>(pos)], cp, pad);
        SyncEv<AscendC::HardEvent::MTE2_S>();
        for (int64_t j = 0; j < n; ++j) {
            int32_t entry = chunk.GetValue(static_cast<int32_t>(j));
            // 越界条目立即报错（替代老路径"塞 owner0 段 + 读表基址行"的静默行为）
            ascendc_assert(entry >= 0 && entry < maxEntry,
                           "rail fwd entry out of range: core=%u row=%lld entry=%d max=%d", aivId_, pos + j, entry,
                           maxEntry);
            int32_t owner = entry / numEntriesPerRank_;
            if (owner != curOwner) {
                // dst 非降序不变量（"每收件区恰一任务"的数学前提）：回退 = 上游被写坏——fail-fast
                ascendc_assert(curOwner < 0 || owner > curOwner,
                               "rail fwd ownerSorted not ascending: core=%u row=%lld owner=%d prev=%d", aivId_, pos + j,
                               owner, curOwner);
                if (curOwner >= 0) {
                    emitRun();
                }
                curOwner = owner;
                runStart = pos + j;
                runLen = 0;
            }
            runLen++;
        }
        pos += n;
    }
    if (curOwner >= 0) {
        emitRun();
    }
    Wr32(segCountsGM_ + (1 + aivId_) * 4, static_cast<int32_t>(runCnt));
}

// S0 合并：同收件区相邻段合并为单一任务（每目的一条流，一区一任务；dst 升序保证同区段连续）。
// accumulateCounts=true 时同步把每条带行数累进 send_counts_out（行程编码分块归约）
__aicore__ inline void EngramFetchTrainRailAlignedArch35::MergeSegmentsBySGroup(bool accumulateCounts)
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
    // 任务按"路由收件区"合并（DIRECT 键=dst 本地号、RAIL 键=dst 超节点号）：一区一任务
    auto areaKey = [&](int32_t dstRank) __attribute__((cce_aicore))->int32_t
    {
        RankCoord c = DecodeRank(topo_, static_cast<uint32_t>(dstRank));
        BucketKind k = ClassifyRow(topo_, me_, static_cast<uint32_t>(dstRank));
        int32_t sub = (k == BUCKET_DIRECT) ? static_cast<int32_t>(c.local) : static_cast<int32_t>(c.supernode);
        return static_cast<int32_t>(k) * RAIL_BUCKET_KEY_STRIDE + sub;
    };
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
            GM_ADDR seg = rdStrip(c, static_cast<uint32_t>(i)); // bypass 读（跨核可见性）
            int32_t dst = Rd32(seg);
            int32_t st = Rd32(seg + 4);
            int32_t len = Rd32(seg + 8);
            if (accumulateCounts) {
                // send_counts_out 累加（每条带恰加一次）：host 已清零（at::zeros）
                GM_ADDR slot = sendCountsOutGM_ + static_cast<uint64_t>(dst) * UB_ALIGN;
                Wr32(slot, Rd32(slot) + len);
            }
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

// 并行两遍前缀（segCount/coreStart 作 scratch）：源 send_counts（32B 槽距），目的 sendPrefix
// （int32，[world]=T）——B2 scatter 摆位 perm[sdispls[src]+k] 的唯一前置
__aicore__ inline void EngramFetchTrainRailAlignedArch35::BuildSendPrefix()
{
    int64_t world = static_cast<int64_t>(topo_.world);
    int64_t per = (world + totalBlocks_ - 1) / totalBlocks_;
    int64_t r0 = static_cast<int64_t>(aivId_) * per;
    int64_t r1 = (r0 + per > world) ? world : r0 + per;
    if (r0 >= world) {
        r0 = world; // 空段不可早退：SyncAll 要求全核到达（8 卡缩比床形态）
        r1 = world;
    }
    int64_t segSum = 0;
    for (int64_t r = r0; r < r1; ++r) {
        segSum += Rd32(sendCountsOutGM_ + r * 32); // send_counts 每项 32B 槽距
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
        base += Rd32(sendCountsOutGM_ + r * 32);
    }
    if (r0 < world && r1 == world) { // 末段核补 T 总量（r0<world 防空段核误写）
        Wr32(sendPrefixGM_ + world * 4, static_cast<int32_t>(base));
    }
}

__aicore__ inline void EngramFetchTrainRailAlignedArch35::BuildRecvPrefixFinal()
{
    int64_t world = static_cast<int64_t>(topo_.world);
    int64_t per = (world + totalBlocks_ - 1) / totalBlocks_;
    int64_t r0 = static_cast<int64_t>(aivId_) * per;
    int64_t r1 = (r0 + per > world) ? world : r0 + per;
    if (r0 >= world) {
        r0 = world;
        r1 = world;
    }
    int64_t segSum = 0;
    for (int64_t r = r0; r < r1; ++r) {
        segSum += Rd32(recvCountsOutGM_ + r * 4); // recv_counts_out 每项 4B
    }
    Wr32(segCountGM_ + aivId_ * 4, static_cast<int32_t>(segSum));
    AscendC::SyncAll<true>();
    if (aivId_ == 0) {
        int64_t base = 0;
        for (uint32_t c = 0; c < totalBlocks_; ++c) {
            Wr32(coreStartGM_ + c * 4, static_cast<int32_t>(base));
            base += Rd32(segCountGM_ + c * 4);
        }
        ascendc_assert(base <= cap_, "rail fwd owner heat overflow: total=%lld cap=%lld", base,
                       static_cast<int64_t>(cap_));
    }
    AscendC::SyncAll<true>();
    int64_t base = Rd32(coreStartGM_ + aivId_ * 4);
    // 逐条 Wr32：连续复用同一 creditBuf_ 时需尾部反向同步，防 MTE3 积压下源字未读走致前缀错表
    for (int64_t r = r0; r < r1; ++r) {
        Wr32(recvPrefixGM_ + r * 4, static_cast<int32_t>(base));
        base += Rd32(recvCountsOutGM_ + r * 4);
    }
    if (r0 < world && r1 == world) {
        Wr32(recvPrefixGM_ + world * 4, static_cast<int32_t>(base));
    }
}

// (src,entry) 追加进本核收集段（每核独立游标，无竞争）；srcArrive[src]++（同源落位 ⇒ 单写者）
__aicore__ inline void EngramFetchTrainRailAlignedArch35::CollectReqRow(int32_t entry, int32_t srcRank)
{
    int32_t arrive = Rd32(srcArriveGM_ + static_cast<uint64_t>(srcRank) * 4);
    Wr32(srcArriveGM_ + static_cast<uint64_t>(srcRank) * 4, arrive + 1);
    ascendc_assert(collectCursor_ < cap_, "rail fwd collect overflow: core=%u cursor=%u cap=%llu src=%d", aivId_,
                   collectCursor_, static_cast<uint64_t>(cap_), srcRank);
    GM_ADDR slot = collectGM_ + (static_cast<uint64_t>(aivId_) * cap_ + collectCursor_) * 8;
    Wr32(slot, srcRank);
    Wr32(slot + 4, entry);
    collectCursor_++;
}

__aicore__ inline void EngramFetchTrainRailAlignedArch35::WriteRecvCounts()
{
    int64_t world = static_cast<int64_t>(topo_.world);
    int64_t per = (world + totalBlocks_ - 1) / totalBlocks_;
    int64_t r0 = static_cast<int64_t>(aivId_) * per;
    int64_t r1 = (r0 + per > world) ? world : r0 + per;
    for (int64_t r = r0; r < r1; ++r) {
        Wr32(recvCountsOutGM_ + r * 4, Rd32(srcArriveGM_ + r * 4));
    }
    SyncEv<AscendC::HardEvent::MTE3_S>();
}

// 每核扫自己收集段（到达序保留——同源行恒经一个区一个核计数）：fill[src]++ ⇒ 稳定散射进
// recv_local_entry[rdispls[src]+k]（段内升序 = 老语义）；srcSorted 伴随落表供 B2 S 用
__aicore__ inline void EngramFetchTrainRailAlignedArch35::StableScatterCollect()
{
    GM_ADDR base = collectGM_ + static_cast<uint64_t>(aivId_) * cap_ * 8;
    for (uint32_t i = 0; i < collectCursor_; ++i) {
        int32_t src = Rd32(base + static_cast<uint64_t>(i) * 8);
        int32_t entry = Rd32(base + static_cast<uint64_t>(i) * 8 + 4);
        int32_t fill = Rd32(srcArriveGM_ + static_cast<uint64_t>(src) * 4);
        Wr32(srcArriveGM_ + static_cast<uint64_t>(src) * 4, fill + 1);
        int32_t pos = Rd32(recvPrefixGM_ + static_cast<uint64_t>(src) * 4) + fill;
        Wr32(recvLocalEntryOutGM_ + static_cast<uint64_t>(pos) * 4, entry);
        Wr32(srcSortedGM_ + static_cast<uint64_t>(pos) * 4, src);
    }
    SyncEv<AscendC::HardEvent::MTE3_S>();
}

__aicore__ inline void EngramFetchTrainRailAlignedArch35::CrossCheckFillAndClear()
{
    int64_t world = static_cast<int64_t>(topo_.world);
    int64_t per = (world + totalBlocks_ - 1) / totalBlocks_;
    int64_t r0 = static_cast<int64_t>(aivId_) * per;
    int64_t r1 = (r0 + per > world) ? world : r0 + per;
    for (int64_t r = r0; r < r1; ++r) {
        int32_t fill = Rd32(srcArriveGM_ + r * 4);
        int32_t cnt = Rd32(recvCountsOutGM_ + r * 4);
        ascendc_assert(fill == cnt, "rail fwd accounting mismatch: src=%lld fill=%d recv_counts=%d", r, fill, cnt);
    }
    AscendC::SyncAll<true>();
    ClearGmRange(srcArriveGM_, r0, r1); // 归零 → B2 每源响应计数
}

// B2 任务条带：每核扫自己 world 切片的 recv_counts（4B），非零源出条带 (dst=src,start=rdispl,cnt)
__aicore__ inline void EngramFetchTrainRailAlignedArch35::BuildRespStrips()
{
    int64_t world = static_cast<int64_t>(topo_.world);
    int64_t per = (world + totalBlocks_ - 1) / totalBlocks_;
    int64_t r0 = static_cast<int64_t>(aivId_) * per;
    int64_t r1 = (r0 + per > world) ? world : r0 + per;
    if (r0 >= world) {
        r0 = world;
        r1 = world;
    }
    GM_ADDR strip = stripsGM_ + static_cast<uint64_t>(aivId_) * stripCap_ * 12;
    uint32_t runCnt = 0;
    for (int64_t r = r0; r < r1; ++r) {
        int32_t cnt = Rd32(recvCountsOutGM_ + r * 4);
        if (cnt <= 0) {
            continue;
        }
        Wr32(strip + runCnt * 12 + 0, static_cast<int32_t>(r));
        Wr32(strip + runCnt * 12 + 4, Rd32(recvPrefixGM_ + r * 4));
        Wr32(strip + runCnt * 12 + 8, cnt);
        runCnt++;
    }
    Wr32(segCountsGM_ + (1 + aivId_) * 4, static_cast<int32_t>(runCnt));
}

__aicore__ inline void EngramFetchTrainRailAlignedArch35::ClearRelayTables()
{
    const uint32_t ranksPerSupernode = topo_.ranksPerSupernode;
    const uint32_t fCnt = (fwdCoreEnd_ > sendCoreEnd_) ? (fwdCoreEnd_ - sendCoreEnd_) : 1U;
    const int64_t rawInts = static_cast<int64_t>(fCnt) * static_cast<int64_t>(ranksPerSupernode);
    const int64_t alignInts = (rawInts * 4 + UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN / 4;
    const int64_t ctlCnt = 3 * alignInts;
    int64_t per = (ctlCnt + totalBlocks_ - 1) / totalBlocks_;
    int64_t s0 = static_cast<int64_t>(aivId_) * per;
    int64_t s1 = (s0 + per > ctlCnt) ? ctlCnt : s0 + per;
    ClearGmRange(relayStagedGM_, s0, s1);
}

// 收件区响应行散位：k=count[src]++（srcArrive 区复用，同源单写者）→ token=perm[sdispls[src]+k]
// → 行数据（+32 跳头）WinLocalCopy 进 fetched（+fetched_sf）
template <bool HasSf>
__aicore__ inline void EngramFetchTrainRailAlignedArch35::ScatterRespRow(GM_ADDR rowGM)
{
    int32_t src = Rd32(rowGM + 4);
    GM_ADDR cntAddr = srcArriveGM_ + static_cast<uint64_t>(src) * 4;
    int32_t k = Rd32(cntAddr);
    Wr32(cntAddr, k + 1);
    int32_t cnt = Rd32(sendCountsOutGM_ + static_cast<uint64_t>(src) * UB_ALIGN);
    ascendc_assert(k < cnt, "rail fwd per-src resp overflow: src=%d k=%d send_counts=%d", src, k, cnt);
    int32_t token =
        Rd32(permOutGM_ + (static_cast<int64_t>(Rd32(sendPrefixGM_ + static_cast<uint64_t>(src) * 4)) + k) * 4);
    ascendc_assert(token >= 0 && token < numTokens_, "rail fwd resp token out of range: token=%d T=%lld", token,
                   numTokens_);
    WinLocalCopy(fetchedGM_ + static_cast<uint64_t>(token) * hiddenBytes_, rowGM + ROW_HEAD_BYTES,
                 static_cast<uint64_t>(hiddenBytes_), tpipe_, entryBuf_.Get<uint8_t>(), gradBuf_.Get<uint8_t>(),
                 evCopy2To3_, evCopy3To2_);
    if constexpr (HasSf) {
        WinLocalCopy(fetchedSfGM_ + static_cast<uint64_t>(token) * sfBytes_, rowGM + ROW_HEAD_BYTES + sfOffBytes_,
                     sfBytes_, tpipe_, entryBuf_.Get<uint8_t>(), gradBuf_.Get<uint8_t>(), evCopy2To3_, evCopy3To2_);
    }
}

// 本地段（src=自己）直读直散：entry → 本地表行 → fetched[perm[sdispls[me]+i]]（i=段内序）。
// count[me]++（srcArrive 区）保持记账闭合——本地行也计入 C2 的 Σ每源响应计数交叉校验
template <bool HasSf>
__aicore__ inline void EngramFetchTrainRailAlignedArch35::ScatterLocalRespRow(int32_t entry, uint32_t arrive)
{
    GM_ADDR cntAddr = srcArriveGM_ + static_cast<uint64_t>(myGlobalRank_) * 4;
    Wr32(cntAddr, Rd32(cntAddr) + 1); // 本地捷径不走 WIN，但响应计数与远端行同口径（断言 2 闭合）
    int32_t local = entry - static_cast<int32_t>(myGlobalRank_) * numEntriesPerRank_;
    ascendc_assert(local >= 0 && local < numEntriesPerRank_, "rail fwd local entry out of range: entry=%d local=%d",
                   entry, local);
    int32_t token =
        Rd32(permOutGM_ + (static_cast<int64_t>(Rd32(sendPrefixGM_ + static_cast<uint64_t>(myGlobalRank_) * 4)) +
                           static_cast<int32_t>(arrive)) *
                              4);
    ascendc_assert(token >= 0 && token < numTokens_, "rail fwd local token out of range: token=%d", token);
    WinLocalCopy(fetchedGM_ + static_cast<uint64_t>(token) * hiddenBytes_,
                 reinterpret_cast<GM_ADDR>(localStorageAddr_) + static_cast<uint64_t>(local) * hiddenBytes_,
                 static_cast<uint64_t>(hiddenBytes_), tpipe_, entryBuf_.Get<uint8_t>(), gradBuf_.Get<uint8_t>(),
                 evCopy2To3_, evCopy3To2_);
    if constexpr (HasSf) {
        WinLocalCopy(fetchedSfGM_ + static_cast<uint64_t>(token) * sfBytes_,
                     reinterpret_cast<GM_ADDR>(sfTableGM_) + static_cast<uint64_t>(local) * sfBytes_, sfBytes_, tpipe_,
                     entryBuf_.Get<uint8_t>(), gradBuf_.Get<uint8_t>(), evCopy2To3_, evCopy3To2_);
    }
}

// ── S 核统一发送：目的 → 属主 S 核固定映射（d<C 为 owner 同伴槽，其余为轨道槽；
//    属主 = d mod sCoreCnt）。自有桶行（B1 拼 REQ 头 / B2 读本地表拼 RESP 块）+ F 暂存
//    转发行同核同流发出；轮转推进（信用满/暂存空让位）；done=writeCnt+1。
/// @brief S 核统一发送循环：每目的一条流，自有桶行 + F 暂存转发行；完结后写 done
/// @tparam IsReq true=req 族（B1），false=resp 族（B2）
/// @return 无（aivId_ 不在 S 组时直接返回）
template <bool IsReq, bool HasSf>
__aicore__ inline void EngramFetchTrainRailAlignedArch35::SendLoopS()
{
    if (aivId_ >= sendCoreEnd_) {
        Wr32(dfxGM_ + aivId_ * 4, 0);
        return;
    }
    const uint32_t sCoreIdx = aivId_;
    const uint32_t sCoreCnt = sendCoreEnd_;
    const uint32_t ranksPerSupernode = topo_.ranksPerSupernode;
    const uint32_t dests = topo_.writerSlots; // ranksPerSupernode + world/ranksPerSupernode（owner 槽 + 轨道槽）
    const uint32_t fCoreCnt = fwdCoreEnd_ - sendCoreEnd_;
    const uint32_t batchRows = IsReq ? batchRowsReq_ : batchRowsResp_;
    const uint32_t P = IsReq ? winReq_.rowStride : winResp_.rowStride;
    const WinLayout& wfam = IsReq ? winReq_ : winResp_;
    const int64_t ringStride = stageRingStride_; // 两族批块取大（req 块 65504B > resp 块）
    // 名下目的数：d ∈ [0,dests) 且 d mod sCoreCnt == 本核（dests < 核数时高编号核名下为空）
    const uint32_t myDests = (sCoreIdx < dests) ? ((dests - 1U - sCoreIdx) / sCoreCnt + 1U) : 0U;
    int32_t taskCnt = Rd32(segCountsGM_);
    auto prog = [&](int32_t code) __attribute__((cce_aicore))
    {
        Wr32(dfxGM_ + static_cast<uint64_t>(aivId_) * 4, code);
    };
    prog(10);

    // 本核目的状态（UB 紧凑布局；下标 k ∈ [0, myDests)）：
    // [0,my)=flowW [my,2my)=flowR [2my,3my)=taskIdx(−1 无任务——0 是合法任务号) [3my,4my)=ownOff [4my,5my)=fin
    LocalTensor<int32_t> sflow = sFlowBuf_.Get<int32_t>();
    for (uint32_t i = 0; i < 5U * myDests; ++i) {
        sflow.SetValue(static_cast<int32_t>(i), 0);
    }
    for (uint32_t k = 0; k < myDests; ++k) {
        sflow.SetValue(static_cast<int32_t>(2 * myDests + k), -1);
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
    // 扫任务表一遍：任务 → 统一目的编号 → 属主核登记（任务按收件区合并，每目的 ≤1 任务）
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
        // 防御：dst 非法（上游产出垃圾）时 dIdx 越界——守卫内嵌（越界写会腐蚀 sflow 邻域 UB）
        if (dIdx < dests && dIdx % sCoreCnt == sCoreIdx) {
            // 同区第二任务 = 条带碎片化：按目的覆盖注册会静默丢行（本地桶例外——O 直读全表）
            int32_t regSlot = static_cast<int32_t>(2 * myDests + dIdx / sCoreCnt);
            int32_t prev = sflow.GetValue(regSlot);
            ascendc_assert(prev < 0 || static_cast<BucketKind>(kindV) == BUCKET_LOCAL,
                           "rail fwd fragmented tasks on dest %u: task=%d shadows prev=%d, rows lost", dIdx, taskIdx,
                           prev);
            sflow.SetValue(regSlot, taskIdx);
        }
    }

    // 目的 → 通道句柄/对端收件区（算术直出；自槽不可达——映射排除；node 表 = S 核目的编号 d 同表直查）
    auto destAddr = [&](uint32_t dIdx, uint64_t & handle, GM_ADDR & area) __attribute__((cce_aicore))
    {
        GM_ADDR ctxBase = reinterpret_cast<GM_ADDR>(ctxHead_);
        handle = NodeChannelAt(ctxBase, dIdx);
        if (dIdx < ranksPerSupernode) { // owner 槽 L'：写同伴 owner 族[myL]
            area = reinterpret_cast<GM_ADDR>(NodeWinAt(ctxBase, dIdx)) + wfam.ownBase +
                   static_cast<uint64_t>(me_.local) * wfam.areaSize;
        } else { // 轨道槽 sn'：写轨道对端（sn' 超节点的同 L 卡）rail 族[mySn]
            area = reinterpret_cast<GM_ADDR>(NodeWinAt(ctxBase, dIdx)) + wfam.railBase +
                   static_cast<uint64_t>(me_.supernode) * wfam.areaSize;
        }
    };

    LocalTensor<uint8_t> block = rowBlockBuf_.Get<uint8_t>();
    LocalTensor<int32_t> head = rowBlockBuf_.Get<int32_t>();
    uint64_t idleSpin = 0;        // 连续无进展轮数（DFX 快照节拍用）
    uint64_t idleT0 = WinNowUs(); // 距上次进展的真墙钟（自旋计数受每转成本影响）
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
            sender.Init(hcomm_, counterScratchGM_ + aivId_ * UB_ALIGN, tpipe_, creditBuf_.Get<int32_t>(), wfam, aivId_,
                        myGlobalRank_, static_cast<uint32_t>(sflow.GetValue(static_cast<int32_t>(k))),
                        sflow.GetValue(static_cast<int32_t>(myDests + k)), dfxGM_);
            bool sent = false;
            // ── ① 自有桶行 ──
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
                    uint32_t rows = static_cast<uint32_t>((taskRows - off > batchRows) ? batchRows : taskRows - off);
                    if (!sender.HasCredit(rows)) {
                        sender.AskCredit(handle, area); // 异步信用读——先转其它目的
                    } else {
                        const uint32_t rowStrideWords = P / 4U; // 行距按 int32 字数（行头填充用）
                        const uint32_t rowHeadBytes = ROW_HEAD_BYTES;
                        if constexpr (IsReq) {
                            // REQ 块：纯 32B 头（dst 逐行算——同区任务可跨多个 dst；
                            // entry 骑头部保留区 +8）
                            for (uint32_t j = 0; j < rows; ++j) {
                                int32_t entry = Rd32(sortedIndicesGM_ + static_cast<uint64_t>(taskStart + off + j) * 4);
                                int32_t rowDst = entry / numEntriesPerRank_;
                                head.SetValue(static_cast<int32_t>(j * rowStrideWords), rowDst);
                                head.SetValue(static_cast<int32_t>(j * rowStrideWords + 1),
                                              static_cast<int32_t>(myGlobalRank_));
                                head.SetValue(static_cast<int32_t>(j * rowStrideWords + 2), entry);
                                head.SetValue(static_cast<int32_t>(j * rowStrideWords + 3), 0);
                            }
                            SyncEv<AscendC::HardEvent::S_MTE3>();
                            GM_ADDR stage = sStageGM_ + static_cast<uint64_t>(sCoreIdx) * ringStride;
                            GlobalTensor<uint8_t> stageGm;
                            stageGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(stage));
                            AscendC::DataCopyExtParams sParams{1U, rows * P, 0U, 0U, 0U};
                            AscendC::DataCopyPad(stageGm, block, sParams);
                            SyncEv<AscendC::HardEvent::MTE3_S>();
                            sender.SendBatch(handle, area, stage, rows);
                            sender.DrainChannel(handle);
                            sflow.SetValue(static_cast<int32_t>(3 * myDests + k), static_cast<int32_t>(off + rows));
                            sent = true;
                        } else {
                            // RESP 块：行头 dst 从 srcSorted 伴随取（同区任务跨多请求者）；
                            // 行数据 = 本地表行（+SF），读入 block 的 +32B 处（UB 侧恒 32B 对齐）
                            GlobalTensor<uint8_t> dataGm;
                            dataGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(localStorageAddr_));
                            AscendC::DataCopyPadExtParams<uint8_t> pad{false, 0, 0, 0};
                            const uint32_t hiddenBytes32 = static_cast<uint32_t>(hiddenBytes_);
                            const uint32_t sfBytes32 = static_cast<uint32_t>(sfBytes_);
                            for (uint32_t j = 0; j < rows; ++j) {
                                int64_t row = taskStart + off + j;
                                int32_t entry = Rd32(recvLocalEntryOutGM_ + static_cast<uint64_t>(row) * 4);
                                int32_t rowDst = Rd32(srcSortedGM_ + static_cast<uint64_t>(row) * 4);
                                int32_t local = entry - static_cast<int32_t>(myGlobalRank_) * numEntriesPerRank_;
                                ascendc_assert(local >= 0 && local < numEntriesPerRank_,
                                               "rail fwd resp entry out of range: entry=%d local=%d", entry, local);
                                head.SetValue(static_cast<int32_t>(j * rowStrideWords), rowDst);
                                head.SetValue(static_cast<int32_t>(j * rowStrideWords + 1),
                                              static_cast<int32_t>(myGlobalRank_));
                                AscendC::DataCopyExtParams dParams{1U, hiddenBytes32, 0U, 0U, 0U};
                                AscendC::DataCopyPad(block[static_cast<uint64_t>(j) * P + rowHeadBytes],
                                                     dataGm[static_cast<uint64_t>(local) * hiddenBytes_], dParams, pad);
                                if constexpr (HasSf) {
                                    GlobalTensor<uint8_t> sfGm;
                                    sfGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(sfTableGM_));
                                    AscendC::DataCopyExtParams sfParams{1U, sfBytes32, 0U, 0U, 0U};
                                    AscendC::DataCopyPad(
                                        block[static_cast<uint64_t>(j) * P + rowHeadBytes + sfOffBytes_],
                                        sfGm[static_cast<uint64_t>(local) * sfBytes_], sfParams, pad);
                                }
                            }
                            SyncEv<AscendC::HardEvent::MTE2_MTE3>();
                            SyncEv<AscendC::HardEvent::S_MTE3>();
                            GM_ADDR stage = sStageGM_ + static_cast<uint64_t>(sCoreIdx) * ringStride;
                            GlobalTensor<uint8_t> stageGm;
                            stageGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(stage));
                            AscendC::DataCopyExtParams sParams{1U, rows * P, 0U, 0U, 0U};
                            AscendC::DataCopyPad(stageGm, block, sParams);
                            SyncEv<AscendC::HardEvent::MTE3_S>();
                            sender.SendBatch(handle, area, stage, rows);
                            sender.DrainChannel(handle);
                            sflow.SetValue(static_cast<int32_t>(3 * myDests + k), static_cast<int32_t>(off + rows));
                            sent = true;
                        }
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
                        uint32_t ringPos = static_cast<uint32_t>(drained) % batchRows;
                        uint32_t fit = batchRows - ringPos; // 源环不跨尾
                        if (run > fit) {
                            run = fit;
                        }
                        if (!sender.HasCredit(run)) {
                            sender.AskCredit(handle, area);
                            break; // 信用不足：先转其它目的/其它 F 核
                        }
                        GM_ADDR src =
                            relayGM_ + idx * static_cast<uint64_t>(ringStride) + static_cast<uint64_t>(ringPos) * P;
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
            sflow.SetValue(static_cast<int32_t>(myDests + k), sender.SyncCredit());
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
                    sender.Finish(handle, area); // Drain + done = writeCnt+1（本地单写）
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
            // 整圈无进展：唯一推进手段是对端 read——自旋完成在途信用读后重转
            prog(20);
            ascendc_assert(WinNowUs() - idleT0 < WIN_TIMEOUT_US, "rail fwd send idle timeout: aiv=%u, unfinished=%u",
                           aivId_, unfinished);
            ++idleSpin;
            // 完结判定失败现场快照（每 DFX_SNAP_REFRESH_IDLE_ROUNDS 轮 idle 刷新一次）：
            // 本核首个未完结 dest 的状态落 flagScratch，任一 RecvLoop 超时随 trap 打出
            if ((idleSpin % DFX_SNAP_REFRESH_IDLE_ROUNDS) == 0U) {
                for (uint32_t k = 0; k < myDests; ++k) {
                    if (sflow.GetValue(static_cast<int32_t>(4 * myDests + k)) != 0) {
                        continue; // 已完结——找首个未完结 dest
                    }
                    uint32_t dIdxS = sCoreIdx + k * sCoreCnt;
                    int32_t tS = sflow.GetValue(static_cast<int32_t>(2 * myDests + k));
                    int64_t offS = sflow.GetValue(static_cast<int32_t>(3 * myDests + k));
                    int64_t cntS = -1;
                    if (tS >= 0) {
                        cntS = Rd32(taskTableGM_ + static_cast<uint64_t>(tS) * 16 + 12);
                    }
                    bool ownDoneS = (tS < 0) || (offS >= cntS);
                    bool allEofS = true;
                    bool drainEqS = true;
                    int32_t badFS = -1;
                    int32_t badStagedS = 0;
                    int32_t badDrainedS = 0;
                    int32_t badEofS = 1;
                    if (dIdxS < ranksPerSupernode && fCoreCnt > 0U) {
                        for (uint32_t f = 0; f < fCoreCnt; ++f) {
                            uint64_t idx = static_cast<uint64_t>(f) * ranksPerSupernode + dIdxS;
                            int32_t stV = Rd32(relayStagedGM_ + idx * 4);
                            int32_t drV = Rd32(relayDrainedGM_ + idx * 4);
                            int32_t eofV = Rd32(relayEofGM_ + idx * 4);
                            if (eofV == 0) {
                                allEofS = false;
                            }
                            if (drV != stV) {
                                drainEqS = false;
                            }
                            if ((eofV == 0 || drV != stV) && badFS < 0) {
                                badFS = static_cast<int32_t>(f);
                                badStagedS = stV;
                                badDrainedS = drV;
                                badEofS = eofV;
                            }
                        }
                    }
                    int32_t tPack = (tS < 0) ? 0 : ((tS > DFX_TASK_IDX_CLAMP) ? DFX_TASK_IDX_CLAMP : tS);
                    int32_t w0S = static_cast<int32_t>(dIdxS) | (ownDoneS ? 1 : 0) << DFX_W0_OWN_DONE_BIT |
                                  (allEofS ? 1 : 0) << DFX_W0_ALL_EOF_BIT | (drainEqS ? 1 : 0) << DFX_W0_DRAIN_EQ_BIT |
                                  (tPack & DFX_TASK_IDX_MASK) << DFX_W0_TASK_SHIFT | 1 << DFX_W0_VALID_BIT;
                    uint64_t snapBase = static_cast<uint64_t>(DFX_SCALAR_SLOTS + sCoreIdx * DFX_SNAP_WORDS) * 4;
                    Wr32(flagScratchGM_ + snapBase, w0S);
                    Wr32(flagScratchGM_ + snapBase + 4, badFS);
                    Wr32(flagScratchGM_ + snapBase + 8, badStagedS);
                    Wr32(flagScratchGM_ + snapBase + 12, badDrainedS | badEofS << DFX_W3_BAD_EOF_SHIFT);
                    break;
                }
            }
            for (uint32_t k = 0; k < myDests; ++k) {
                if (sflow.GetValue(static_cast<int32_t>(4 * myDests + k)) != 0) {
                    continue;
                }
                uint32_t dIdx = sCoreIdx + k * sCoreCnt;
                uint64_t handle = 0;
                GM_ADDR area = nullptr;
                destAddr(dIdx, handle, area);
                RowSender sender;
                sender.Init(hcomm_, counterScratchGM_ + aivId_ * UB_ALIGN, tpipe_, creditBuf_.Get<int32_t>(), wfam,
                            aivId_, myGlobalRank_, static_cast<uint32_t>(sflow.GetValue(static_cast<int32_t>(k))),
                            sflow.GetValue(static_cast<int32_t>(myDests + k)), dfxGM_);
                sender.WaitCredit(handle, area);
                sflow.SetValue(static_cast<int32_t>(myDests + k), sender.RemoteReadCnt());
            }
        } else {
            idleSpin = 0;
            idleT0 = WinNowUs(); // 有进展——"距上次进展"计时复位
        }
    }
    prog(29);
    // 全目的完结——清本核完结判定快照槽（RecvLoop 超时里的 stallCnt = 未完结核数）
    Wr32(flagScratchGM_ + static_cast<uint64_t>(DFX_SCALAR_SLOTS + sCoreIdx * DFX_SNAP_WORDS) * 4, 0);
}

// ── F 核：纯消费 + 暂存（不碰通道）。按行头 dst 分类：dst==self 本地终点；否则暂存
//    owner 槽由属主 S 核直投；rail 区收齐即置 EOF。
/// @brief F 核纯消费循环：单相消费 rail 区
/// @return 无（aivId_ 不在 F 组时直接返回）
template <bool IsReq, bool HasSf>
__aicore__ inline void EngramFetchTrainRailAlignedArch35::FwdLoopF()
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
    const uint32_t batchRows = IsReq ? batchRowsReq_ : batchRowsResp_;
    const uint32_t P = IsReq ? winReq_.rowStride : winResp_.rowStride;
    const WinLayout& wfam = IsReq ? winReq_ : winResp_;
    const int64_t ringStride = stageRingStride_; // 两族批块取大（req 块 65504B > resp 块）
    auto prog = [&](int32_t code) __attribute__((cce_aicore))
    {
        Wr32(dfxGM_ + static_cast<uint64_t>(aivId_) * 4, code);
    };

    LocalTensor<int32_t> staged = flowBuf_.Get<int32_t>();
    for (uint32_t i = 0; i < slots; ++i) {
        staged.SetValue(static_cast<int32_t>(i), 0);
    }
    auto relaySlot = [&](uint32_t slot) __attribute__((cce_aicore))
    {
        return relayGM_ + (static_cast<uint64_t>(fCoreIdx) * slots + slot) * static_cast<uint64_t>(ringStride);
    };
    auto stageRowReq = [&](GM_ADDR rowGM, uint32_t slot) __attribute__((cce_aicore))
    {
        uint64_t idx = static_cast<uint64_t>(fCoreIdx) * slots + slot;
        int32_t n = staged.GetValue(static_cast<int32_t>(slot));
        // 背压：环满（staged − drained ≥ 批行数）等属主 S 核排空再复写该行格（真墙钟 60s 兜底）
        uint64_t t0 = WinNowUs();
        while (n - Rd32(relayDrainedGM_ + idx * 4) >= static_cast<int32_t>(batchRows)) {
            ascendc_assert(WinNowUs() - t0 < WIN_TIMEOUT_US,
                           "rail fwd relay backpressure timeout: f=%u, slot=%u, staged=%d", fCoreIdx, slot, n);
        }
        GM_ADDR dstSlot = relaySlot(slot) + static_cast<uint64_t>(n % batchRows) * P;
        Wr32(dstSlot + 0, Rd32(rowGM + 0)); // 行原样 4×int32（dst/src/entry/rsvd）——req 行恒 32B
        Wr32(dstSlot + 4, Rd32(rowGM + 4));
        Wr32(dstSlot + 8, Rd32(rowGM + 8));
        Wr32(dstSlot + 12, Rd32(rowGM + 12));
        staged.SetValue(static_cast<int32_t>(slot), n + 1);
        Wr32(relayStagedGM_ + idx * 4, n + 1); // 行数据（MTE3）落地后才发布计数——Wr32 同款顺序保证
    };
    auto stageRowResp = [&](GM_ADDR rowGM, uint32_t slot) __attribute__((cce_aicore))
    {
        uint64_t idx = static_cast<uint64_t>(fCoreIdx) * slots + slot;
        int32_t n = staged.GetValue(static_cast<int32_t>(slot));
        uint64_t t0 = WinNowUs();
        while (n - Rd32(relayDrainedGM_ + idx * 4) >= static_cast<int32_t>(batchRows)) {
            ascendc_assert(WinNowUs() - t0 < WIN_TIMEOUT_US,
                           "rail fwd relay backpressure timeout: f=%u, slot=%u, staged=%d", fCoreIdx, slot, n);
        }
        WinLocalCopy(relaySlot(slot) + static_cast<uint64_t>(n % batchRows) * P, rowGM, P, tpipe_,
                     entryBuf_.Get<uint8_t>(), gradBuf_.Get<uint8_t>(), evCopy2To3_, evCopy3To2_);
        staged.SetValue(static_cast<int32_t>(slot), n + 1);
        Wr32(relayStagedGM_ + idx * 4, n + 1);
    };
    // 统一分类：一行进槽（行原样暂存；攒批/信用/远程写全部由属主 S 核接管）
    auto fwdRow = [&](GM_ADDR rowGM) __attribute__((cce_aicore))
    {
        uint32_t dst = static_cast<uint32_t>(Rd32(rowGM));
        if (dst == myGlobalRank_) { // 轨道对端即终点（dst==self）→ 直接本地终点
            if constexpr (IsReq) {
                CollectReqRow(Rd32(rowGM + 8), Rd32(rowGM + 4)); // (entry, src)
            } else {
                ScatterRespRow<HasSf>(rowGM);
            }
            return;
        }
        uint32_t ownerLocal = 0;
        (void)FwdRoute(topo_, me_, dst, ownerLocal); // 轨道区行 dst 必在本超节点
        if constexpr (IsReq) {
            stageRowReq(rowGM, ownerLocal); // 暂存 owner 槽（键 = 其本地号 L_T）
        } else {
            stageRowResp(rowGM, ownerLocal);
        }
    };
    auto publishEof = [&](uint32_t fromSlot, uint32_t toSlot) __attribute__((cce_aicore))
    {
        for (uint32_t s = fromSlot; s < toSlot; ++s) {
            Wr32(relayEofGM_ + (static_cast<uint64_t>(fCoreIdx) * slots + s) * 4, 1);
        }
    };

    // ── 单相：rail 族（写者 = 各超节点同 L 卡的 S 轨道流）──
    prog(60);
    for (uint32_t sn = fCoreIdx; sn < topo_.supernodeCnt; sn += fCoreCnt) {
        if (sn == me_.supernode) {
            continue; // 本超节点一格空置（无写者）
        }
        RecvLoop(
            RailArea(localWin_, wfam, sn), wfam, *tpipe_, creditBuf_.Get<int32_t>(),
            [&](GM_ADDR blockGM, uint32_t rows) __attribute__((cce_aicore)) {
                for (uint32_t j = 0; j < rows; ++j) {
                    fwdRow(blockGM + j * P);
                }
            },
            dfxGM_, flagScratchGM_, sendCoreEnd_);
    }
    publishEof(0, ranksPerSupernode); // owner 槽仅本相供行——rail 族收齐即 EOF
    prog(63);
}

// ── O 核：B1 轮询 owner-req 区做 (src,entry) 记账；B2 轮询 owner-resp 区散位进 fetched。
//    落位由协议决定，不按核分段。
/// @brief O 核收集循环：轮询名下 owner 区 + 本地桶任务
/// @return 无（aivId_ 不在 O 组时直接返回）
template <bool IsReq, bool HasSf>
__aicore__ inline void EngramFetchTrainRailAlignedArch35::OwnerLoopO()
{
    if (aivId_ < fwdCoreEnd_) {
        return;
    }
    uint32_t oCoreIdx = aivId_ - fwdCoreEnd_;
    uint32_t oCoreCnt = totalBlocks_ - fwdCoreEnd_;
    if (oCoreCnt == 0U) {
        return;
    }
    const uint32_t ranksPerSupernode = topo_.ranksPerSupernode;
    const uint32_t P = IsReq ? winReq_.rowStride : winResp_.rowStride;
    const WinLayout& wfam = IsReq ? winReq_ : winResp_;
    auto prog = [&](int32_t code) __attribute__((cce_aicore))
    {
        Wr32(dfxGM_ + static_cast<uint64_t>(aivId_) * 4, code);
    };
    prog(70);
    // 名下 owner 收件区轮转（L' = 写者本地号；ranksPerSupernode 格，自槽 myL 空置）
    for (uint32_t peerLocal = oCoreIdx; peerLocal < ranksPerSupernode; peerLocal += oCoreCnt) {
        if (peerLocal == me_.local) {
            continue;
        }
        RecvLoop(
            OwnArea(localWin_, wfam, peerLocal), wfam, *tpipe_, creditBuf_.Get<int32_t>(),
            [&](GM_ADDR blockGM, uint32_t rows) __attribute__((cce_aicore)) {
                for (uint32_t j = 0; j < rows; ++j) {
                    GM_ADDR rowGM = blockGM + j * P;
                    if constexpr (IsReq) {
                        // REQ 行：+8 = entry（骑行头保留区）
                        CollectReqRow(Rd32(rowGM + 8), Rd32(rowGM + 4));
                    } else {
                        ScatterRespRow<HasSf>(rowGM);
                    }
                }
            },
            dfxGM_, flagScratchGM_, sendCoreEnd_);
    }
    // 本地桶任务（dst==myRank）：B1 行取自 sortedIndices 无通信；B2 行取自 recv_local_entry 段
    int32_t taskCnt = Rd32(segCountsGM_);
    for (int32_t taskIdx = static_cast<int32_t>(oCoreIdx); taskIdx < taskCnt;
         taskIdx += static_cast<int32_t>(oCoreCnt)) {
        GM_ADDR taskG = taskTableGM_ + static_cast<uint64_t>(taskIdx) * 16;
        if (Rd32(taskG) != BUCKET_LOCAL) {
            continue;
        }
        int64_t startRow = Rd32(taskG + 8);
        int64_t cnt = Rd32(taskG + 12);
        for (int64_t r = 0; r < cnt; ++r) {
            if constexpr (IsReq) {
                CollectReqRow(Rd32(sortedIndicesGM_ + static_cast<uint64_t>(startRow + r) * 4),
                              static_cast<int32_t>(myGlobalRank_));
            } else {
                ScatterLocalRespRow<HasSf>(Rd32(recvLocalEntryOutGM_ + static_cast<uint64_t>(startRow + r) * 4),
                                           static_cast<uint32_t>(r));
            }
        }
    }
    prog(71);
}

__aicore__ inline void EngramFetchTrainRailAlignedArch35::SumRespCounts()
{
    int64_t world = static_cast<int64_t>(topo_.world);
    int64_t per = (world + totalBlocks_ - 1) / totalBlocks_;
    int64_t r0 = static_cast<int64_t>(aivId_) * per;
    int64_t r1 = (r0 + per > world) ? world : r0 + per;
    int32_t segSum = 0;
    for (int64_t r = r0; r < r1; ++r) {
        segSum += Rd32(srcArriveGM_ + r * 4);
    }
    Wr32(segCountGM_ + aivId_ * 4, segSum);
}

template <bool HasSf>
__aicore__ inline void EngramFetchTrainRailAlignedArch35::Process()
{
    if (g_coreType != AscendC::AIV) {
        return;
    }
    InitCoreGroups();
    collectCursor_ = 0;

    uint32_t epoch = static_cast<uint32_t>(
        Rd32(localWin_ + static_cast<uint64_t>(topo_.writerSlots) * STATE_OFFSET)); // = done(上轮) = 2c
    CheckWinHeadInvariant(epoch);  // 窗头入参一致性（违例即 trap）
    BarrierWaitWritersDone(epoch); // 等全部写者上轮收尾（首次调用 epoch=0 直通）
    ClearRecvAreas();              // 2W 收件区头 + srcArrive + relay 三表清零（分片）
    AscendC::SyncAll<true>();
    BarrierAnnounceCleared(epoch + 1U); // 通告"已清可写"（clr(c)=2c+1），写者收到才许写我

    SortInputIndicesRail();
    InitPipeAndBuffers(); // 排序后重建常驻通信缓冲（老 train 同款时序）
    AscendC::SyncAll<true>();
    BucketClassify();
    AscendC::SyncAll<true>();
    MergeSegmentsBySGroup(true); // S0 合并 → B1 任务表 + send_counts 累加
    AscendC::SyncAll<true>();
    BuildSendPrefix(); // sdispls（含内部两遍 SyncAll）
    AscendC::SyncAll<true>();
    BarrierWaitPeersCleared(epoch + 1U); // 发送门：写者们本轮"已清可写"全部就位

    if (aivId_ < sendCoreEnd_) {
        SendLoopS<true, HasSf>();
    } else if (aivId_ < fwdCoreEnd_) {
        FwdLoopF<true, HasSf>();
    } else {
        OwnerLoopO<true, HasSf>();
    }
    AscendC::SyncAll<true>();

    WriteRecvCounts(); // recv_counts[r] = srcArrive[r]
    AscendC::SyncAll<true>();
    {
        // 归零 → fill[]（分片）
        int64_t world = static_cast<int64_t>(topo_.world);
        int64_t per = (world + totalBlocks_ - 1) / totalBlocks_;
        int64_t s0 = static_cast<int64_t>(aivId_) * per;
        int64_t s1 = (s0 + per > world) ? world : s0 + per;
        ClearGmRange(srcArriveGM_, s0, s1);
    }
    AscendC::SyncAll<true>();
    BuildRecvPrefixFinal(); // rdispls + [world]=Σ（含热度上限守卫）
    AscendC::SyncAll<true>();
    StableScatterCollect(); // 收集数组稳定散射 → recv_local_entry(+srcSorted)
    AscendC::SyncAll<true>();
    CrossCheckFillAndClear(); // fill==recv_counts 交叉校验；srcArrive 再清（B2 count）
    AscendC::SyncAll<true>();
    BuildRespStrips(); // 非零源条带（world 切片）
    AscendC::SyncAll<true>();
    MergeSegmentsBySGroup(false); // S0 合并 → B2 任务表
    AscendC::SyncAll<true>();
    ClearRelayTables(); // 暂存三表重清（B2 复用；环必空——B1 全收齐）
    AscendC::SyncAll<true>();

    if (aivId_ < sendCoreEnd_) {
        SendLoopS<false, HasSf>();
    } else if (aivId_ < fwdCoreEnd_) {
        FwdLoopF<false, HasSf>();
    } else {
        OwnerLoopO<false, HasSf>();
    }
    AscendC::SyncAll<true>();

    SumRespCounts();
    AscendC::SyncAll<true>();
    if (aivId_ == 0) {
        int64_t total = 0;
        for (uint32_t c = 0; c < totalBlocks_; ++c) {
            total += Rd32(segCountGM_ + c * 4);
        }
        int32_t tSum = Rd32(sendPrefixGM_ + static_cast<int64_t>(topo_.world) * 4);
        // 断言 2：请求方 Σ每源响应计数 == numTokens == Σsend_counts
        ascendc_assert(total == tSum && total == numTokens_,
                       "rail fwd cross-check failed: total=%lld T-sum=%d numTokens=%lld", total, tSum, numTokens_);
        Wr32(flagScratchGM_, static_cast<int32_t>(total));
        // num_recv = Σ recv_counts（owner 侧实收总数 = 反向记账的期望接收总量）
        int32_t numRecv = Rd32(recvPrefixGM_ + static_cast<int64_t>(topo_.world) * 4);
        Wr32(numRecvOutGM_, numRecv);
    }
    AscendC::SyncAll<true>();
    BarrierWriteCallDone(epoch + 2U);
}

} // namespace Mc2Kernel

#endif // ENABLE_ENGRAM_FETCH_KERNEL

#endif // ENGRAM_FETCH_TRAIN_RAILALIGNED_ARCH35_H
