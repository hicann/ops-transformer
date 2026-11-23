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
 * \file engram_fetch_grad_win.h
 * \brief 分轨（rail-aligned）WIN 区布局、收件区寻址与行收发原语，自含不依赖老通信头。
 *        WIN = 窗头 | rail 区 | owner 区；收件区 M = 96B 三计数头 + N 个行格（步长 P，环形复用）。
 *        窗头 = (W+1)×32B（W 个 barrier 写者槽 + 1 个 epoch 槽）；零映射表，格号算术直出。
 */

#ifndef ENGRAM_FETCH_GRAD_WIN_H
#define ENGRAM_FETCH_GRAD_WIN_H

#include "engram_fetch_grad_route.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "adv_api/hccl/hccl.h"

namespace Mc2Kernel {

constexpr uint64_t WIN_TIMEOUT_US = 60ULL * 1000ULL * 1000ULL;
// DFX 取样偏移（字节，dfxSeg 内）：F 进度 0/1 与 O 进度 0/1（超时取证样本）
constexpr uint32_t DFX_F_PROG0_OFF = 48U;
constexpr uint32_t DFX_F_PROG1_OFF = 52U;
constexpr uint32_t DFX_O_PROG0_OFF = 152U;
constexpr uint32_t DFX_O_PROG1_OFF = 156U;
// 超时取证最多判读的 S 核快照数（ascendc_assert 参数预算）
constexpr uint32_t DFX_STALL_SAMPLE_MAX = 12U;
// S 核完结判定失败快照的刷新节拍（连续无进展轮数）
constexpr uint64_t DFX_SNAP_REFRESH_IDLE_ROUNDS = 256ULL;
// flagScratch DFX 快照区（fwd 写 / RecvLoop 读）：[0,16)=标量槽，其后每核 4 字快照。
// w0 位域：b0..4=首未完结 dIdx、b5=ownDone、b6=allEof、b7=drainEq、b8..17=任务号(10bit)、b31=有效
constexpr uint32_t DFX_SCALAR_SLOTS = 16U;
constexpr uint32_t DFX_SNAP_WORDS = 4U;
constexpr uint32_t DFX_W0_OWN_DONE_BIT = 5U;
constexpr uint32_t DFX_W0_ALL_EOF_BIT = 6U;
constexpr uint32_t DFX_W0_DRAIN_EQ_BIT = 7U;
constexpr uint32_t DFX_W0_TASK_SHIFT = 8U;
constexpr int32_t DFX_TASK_IDX_CLAMP = 1023; // 任务号 10bit 钳制（饱和值）
constexpr uint32_t DFX_TASK_IDX_MASK = 0x3FFU;
constexpr uint32_t DFX_W0_VALID_BIT = 31U;
constexpr uint32_t DFX_W3_BAD_EOF_SHIFT = 16U;

__aicore__ inline uint64_t WinNowUs()
{
    return static_cast<uint64_t>(AscendC::GetSystemCycle()) / 1000ULL; // 1kHz cycle → us
}

template <AscendC::HardEvent event>
__aicore__ inline void EngramWinSyncFunc(AscendC::TPipe& pipe)
{
    int32_t eventID = static_cast<int32_t>(pipe.FetchEventID(event));
    AscendC::SetFlag<event>(eventID);
    AscendC::WaitFlag<event>(eventID);
}

struct WinLayout {
    uint32_t rowStride; // P = align32(32 + hiddenBytes)，行 = 32B 行头 + 数据
    uint64_t areaSize;  // M = 96 + N×P（单个收件区，三计数头）
    uint32_t numRows;   // N：在途上限（行/来源，commBufferSize 分满反解，非 2 的幂）
    RailTopo topo;      // 运行期拓扑（ranksPerSupernode 感知 + 超节点数/轨道数推导）
    uint64_t winHead;   // = (W + 1) × 32（W 个 barrier 写者槽 + 1 个 epoch 槽）
    uint64_t railBase;  // rail 区基址 = winHead（轨道写者，按源超节点号 sn' 直排）
    uint64_t ownBase;   // owner 区基址
};

/// @brief 推导 WIN 布局：行距/收件区尺寸/窗头/rail|owner 两区基址
__aicore__ inline WinLayout DeriveWinLayout(const RailTopo& topo, int64_t hiddenBytes, uint32_t numRows)
{
    WinLayout w;
    w.topo = topo;
    const int64_t rowStrideI64 = (hiddenBytes + ROW_HEAD_BYTES + UB_ALIGN - 1) / UB_ALIGN * UB_ALIGN;
    w.rowStride = static_cast<uint32_t>(rowStrideI64);
    ascendc_assert(static_cast<int64_t>(w.rowStride) == rowStrideI64,
                   "rail win rowStride overflow: P=%lld (hiddenBytes=%lld)", rowStrideI64, hiddenBytes);
    w.numRows = numRows;
    w.areaSize = RECV_AREA_HEAD + static_cast<uint64_t>(numRows) * w.rowStride;
    w.winHead = static_cast<uint64_t>(topo.writerSlots + EPOCH_SLOT_TAIL) * STATE_OFFSET;
    w.railBase = w.winHead;
    w.ownBase = w.railBase + static_cast<uint64_t>(topo.supernodeCnt) * w.areaSize;
    return w;
}

/// @brief 取本卡 rail 收件区地址：按源超节点号 sn' 直排（本超节点一格空置）
__aicore__ inline GM_ADDR RailArea(GM_ADDR win, const WinLayout& w, uint32_t srcSupernode)
{
    return win + w.railBase + static_cast<uint64_t>(srcSupernode) * w.areaSize;
}
/// @brief 取本卡 owner 收件区地址：按写者本地号 L 直排（自己一格空置）
__aicore__ inline GM_ADDR OwnArea(GM_ADDR win, const WinLayout& w, uint32_t writerLocal)
{
    return win + w.ownBase + static_cast<uint64_t>(writerLocal) * w.areaSize;
}
/// @brief 取收件区内第 t 个行格地址（t = 计数 mod N，环形复用）
__aicore__ inline GM_ADDR RowSlot(GM_ADDR area, const WinLayout& w, uint32_t t)
{
    return area + RECV_AREA_HEAD + static_cast<uint64_t>(t) * w.rowStride;
}

// ══════════════════ 三计数读写 ══════════════════
__aicore__ inline int32_t ReadCnt(GM_ADDR area, uint32_t off)
{
    return AscendC::ReadGmBypassDCache(reinterpret_cast<__gm__ int32_t*>(area + off));
}

/// @brief 判定收件区是否完结
__aicore__ inline bool AreaFinished(GM_ADDR area, int32_t received)
{
    int32_t done = ReadCnt(area, CNT_DONE_OFF);
    return done != 0 && received == done - 1;
}

/// @brief 回写本卡收件区 read 计数（发送方远程读它当信用）：UB 填值 → GM 直写 32B
__aicore__ inline void WriteLocalReadCnt(GM_ADDR area, uint32_t value, AscendC::TPipe& pipe,
                                         AscendC::LocalTensor<int32_t> statusUb)
{
    statusUb.SetValue(0, static_cast<int32_t>(value));
    EngramWinSyncFunc<AscendC::HardEvent::S_MTE3>(pipe);
    AscendC::GlobalTensor<int32_t> cntGM;
    cntGM.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(area + CNT_READ_OFF));
    AscendC::DataCopyParams params = {1U, static_cast<uint16_t>(UB_ALIGN), 0U, 0U};
    AscendC::DataCopyPad(cntGM, statusUb, params);
    EngramWinSyncFunc<AscendC::HardEvent::MTE3_S>(pipe);
}

// 行发送器：管理与单一目的的行流状态；信用窗口约束在途 = writeCnt − remoteReadCnt ≤ N
class RowSender {
public:
    __aicore__ inline RowSender() = default;

    // scratch：本核独占 32B 信用读落地格；initWriteCnt/initRemoteRead：流状态由调用方持久
    // 保存、每批装载/读回（每目的一条流，轮转装载复用）；dfxSeg：进度码区（可选，取证用）
    __aicore__ inline void Init(AscendC::Hcomm<AscendC::COMM_PROTOCOL_UBC_CTP>& comm, GM_ADDR scratch,
                                AscendC::TPipe* pipe, AscendC::LocalTensor<int32_t> statusUb, const WinLayout& w,
                                uint32_t aivId, uint32_t rankId, uint32_t initWriteCnt = 0, int32_t initRemoteRead = 0,
                                GM_ADDR dfxSeg = nullptr)
    {
        comm_ = &comm;
        scratch_ = scratch;
        pipe_ = pipe;
        statusUb_ = statusUb;
        w_ = w;
        aivId_ = aivId;
        rankId_ = rankId;
        writeCnt_ = initWriteCnt;
        remoteReadCnt_ = initRemoteRead;
        creditInFlight_ = false;
        dfxSeg_ = dfxSeg;
    }

    // ── 多目的轮转的非阻塞信用 API（轮转不因单一目的信用满而停摆）──
    // 信用是否够本批（在途 = writeCnt_ − remoteReadCnt_ 须 ≤ N）
    __aicore__ inline bool HasCredit(uint32_t rows) const
    {
        return writeCnt_ + rows - static_cast<uint32_t>(remoteReadCnt_) <= w_.numRows;
    }
    // 异步发起信用读（在途则跳过；非阻塞——S 核用它后继续轮转其它目的）
    __aicore__ inline void AskCredit(uint64_t handle, GM_ADDR remoteArea)
    {
        PrefetchCredit(handle, remoteArea);
    }
    // 自旋等信用读落地并刷新（仅在整圈无进展时调用——此时唯一推进手段就是对端 read）
    __aicore__ inline void WaitCredit(uint64_t handle, GM_ADDR remoteArea)
    {
        PrefetchCredit(handle, remoteArea);
        remoteReadCnt_ = CompleteCredit();
    }
    // 收编在途信用读（目的切换前必须调用，防两目的交错写同一 scratch 格致信用值错配）
    __aicore__ inline int32_t SyncCredit()
    {
        if (creditInFlight_) {
            remoteReadCnt_ = CompleteCredit();
        }
        return remoteReadCnt_;
    }
    // 等本通道全部在途写落地（WriteNbi 非阻塞，源 GM 被取走前不得复写）
    __aicore__ inline void DrainChannel(uint64_t handle)
    {
        Drain(handle);
    }

    // 写一批行（blockGM = 逐行 32B 头 + 数据）：环形行格跨尾拆两段，write 计数随末段
    // WriteWithNotifyNbi 更新；信用满则先远程读对端 read 计数等信用
    __aicore__ inline void SendBatch(uint64_t handle, GM_ADDR remoteArea, GM_ADDR blockGM, uint32_t rows)
    {
        while (writeCnt_ + rows - static_cast<uint32_t>(remoteReadCnt_) > w_.numRows) {
            PrefetchCredit(handle, remoteArea);
            remoteReadCnt_ = CompleteCredit(); // 自旋等 scratch 非 sentinel（60s 超时）
        }
        uint32_t t0 = writeCnt_ % w_.numRows;
        uint32_t fit = w_.numRows - t0;
        if (rows <= fit) {
            WriteWithNotify(handle, RowSlot(remoteArea, w_, t0), blockGM, static_cast<uint64_t>(rows) * w_.rowStride,
                            remoteArea + CNT_WRITE_OFF, writeCnt_ + rows);
        } else {
            WriteNbi(handle, RowSlot(remoteArea, w_, t0), blockGM, static_cast<uint64_t>(fit) * w_.rowStride);
            WriteWithNotify(handle, RowSlot(remoteArea, w_, 0), blockGM + static_cast<uint64_t>(fit) * w_.rowStride,
                            static_cast<uint64_t>(rows - fit) * w_.rowStride, remoteArea + CNT_WRITE_OFF,
                            writeCnt_ + rows);
        }
        writeCnt_ += rows;
    }

    // 流完结：排空 + done = writeCnt+1（+1 偏置区分"未写"；属主 S 核单写）
    __aicore__ inline void Finish(uint64_t handle, GM_ADDR remoteArea)
    {
        if (creditInFlight_) {
            (void)CompleteCredit();
        }
        Drain(handle);
        WriteCounter32(handle, remoteArea + CNT_DONE_OFF, static_cast<int32_t>(writeCnt_ + 1U));
        Drain(handle);
    }

    __aicore__ inline uint32_t WriteCnt() const
    {
        return writeCnt_;
    }
    __aicore__ inline int32_t RemoteReadCnt() const
    {
        return remoteReadCnt_;
    }

private:
    // 非阻塞信用读：sentinel 先落 scratch → ReadNbi 覆盖为对端 read 计数
    __aicore__ inline void PrefetchCredit(uint64_t handle, GM_ADDR remoteArea)
    {
        if (creditInFlight_) {
            return;
        }
        remoteArea_ = remoteArea; // 记录最近一次信用读目标（取证补读用）
        lastHandle_ = handle;
        statusUb_.SetValue(0, CREDIT_SENTINEL); // sentinel 先落 scratch
        EngramWinSyncFunc<AscendC::HardEvent::S_MTE3>(*pipe_);
        AscendC::GlobalTensor<int32_t> scratchGM;
        scratchGM.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(scratch_));
        AscendC::DataCopyParams sentinelParams = {1U, static_cast<uint16_t>(UB_ALIGN), 0U, 0U};
        AscendC::DataCopyPad(scratchGM, statusUb_, sentinelParams);
        EngramWinSyncFunc<AscendC::HardEvent::MTE3_S>(*pipe_);

        GM_ADDR remoteCounterAddr = remoteArea + CNT_READ_OFF;
        int32_t ret = comm_->ReadNbi(handle, scratch_, remoteCounterAddr, sizeof(int32_t));
        ascendc_assert(ret == 0, "ReadNbi(credit) failed, ret=%d, rankId=%u, aivId=%u", ret, rankId_, aivId_);
        creditInFlight_ = true;
    }

    __aicore__ inline int32_t CompleteCredit()
    {
        AscendC::GlobalTensor<int32_t> scratchGM;
        scratchGM.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(scratch_));
        int32_t value = CREDIT_SENTINEL;
        uint64_t t0 = WinNowUs(); // 真墙钟 60s
        while (value == CREDIT_SENTINEL) {
            AscendC::DataCopyExtParams cpParams{1U, UB_ALIGN, 0U, 0U, 0U};
            AscendC::DataCopyPadExtParams<int32_t> cpPad{false, 0, 0, 0};
            AscendC::DataCopyPad(statusUb_, scratchGM, cpParams, cpPad);
            EngramWinSyncFunc<AscendC::HardEvent::MTE2_S>(*pipe_);
            value = statusUb_.GetValue(0);
            // 读已落地但值 == 0xFFFFFFFF：对端该收件区 read 计数是"从未清零/回写"的
            // 垃圾（对端挂死在两轮门未清区，或窗被集合 scratch 污染）
            if (value == CREDIT_CNT_GARBAGE) {
                TrapCreditStuck(true);
            }
            if (WinNowUs() - t0 > WIN_TIMEOUT_US) {
                TrapCreditStuck(false);
            }
        }
        creditInFlight_ = false;
        return value;
    }

    // 信用读卡死取证：补读对端 write/done 计数 + 各组进度码，一次 trap 打全
    __aicore__ inline void TrapCreditStuck(bool garbage)
    {
        int32_t peerWrite = ReadPeerCntForDump(CNT_WRITE_OFF);
        int32_t peerDone = ReadPeerCntForDump(CNT_DONE_OFF);
        if (garbage) {
            ascendc_assert(false,
                           "credit read GARBAGE: peer readCnt==-1 (stale area head): rankId=%u, aivId=%u, "
                           "writeCnt=%u, remoteRead=%d, peerWrite=%d, peerDone=%d, dfxS0=%d, dfxF12=%d, "
                           "dfxF13=%d, dfxO38=%d, dfxO39=%d",
                           rankId_, aivId_, writeCnt_, remoteReadCnt_, peerWrite, peerDone, DfxAt(0U),
                           DfxAt(DFX_F_PROG0_OFF), DfxAt(DFX_F_PROG1_OFF), DfxAt(DFX_O_PROG0_OFF),
                           DfxAt(DFX_O_PROG1_OFF));
        }
        ascendc_assert(false,
                       "credit read TIMEOUT: read never landed (scratch==sentinel): rankId=%u, aivId=%u, "
                       "writeCnt=%u, remoteRead=%d, peerWrite=%d, peerDone=%d, dfxS0=%d, dfxF12=%d, "
                       "dfxF13=%d, dfxO38=%d, dfxO39=%d",
                       rankId_, aivId_, writeCnt_, remoteReadCnt_, peerWrite, peerDone, DfxAt(0U),
                       DfxAt(DFX_F_PROG0_OFF), DfxAt(DFX_F_PROG1_OFF), DfxAt(DFX_O_PROG0_OFF), DfxAt(DFX_O_PROG1_OFF));
    }

    // 补读对端另一计数（write/done）：sentinel 占位 → ReadNbi → 短自旋等值
    __aicore__ inline int32_t ReadPeerCntForDump(uint32_t off)
    {
        if (remoteArea_ == nullptr || lastHandle_ == 0U) {
            return CREDIT_SENTINEL;
        }
        statusUb_.SetValue(0, CREDIT_SENTINEL);
        EngramWinSyncFunc<AscendC::HardEvent::S_MTE3>(*pipe_);
        AscendC::GlobalTensor<int32_t> scratchGM;
        scratchGM.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(scratch_));
        AscendC::DataCopyParams params = {1U, static_cast<uint16_t>(UB_ALIGN), 0U, 0U};
        AscendC::DataCopyPad(scratchGM, statusUb_, params);
        EngramWinSyncFunc<AscendC::HardEvent::MTE3_S>(*pipe_);
        int32_t ret = comm_->ReadNbi(lastHandle_, scratch_, remoteArea_ + off, sizeof(int32_t));
        if (ret != 0) {
            return CREDIT_SENTINEL; // 补读失败按"未落地"口径呈现
        }
        int32_t value = CREDIT_SENTINEL;
        uint64_t spin = 0;
        while (value == CREDIT_SENTINEL) {
            AscendC::DataCopyExtParams cpParams{1U, UB_ALIGN, 0U, 0U, 0U};
            AscendC::DataCopyPadExtParams<int32_t> cpPad{false, 0, 0, 0};
            AscendC::DataCopyPad(statusUb_, scratchGM, cpParams, cpPad);
            EngramWinSyncFunc<AscendC::HardEvent::MTE2_S>(*pipe_);
            value = statusUb_.GetValue(0);
            if (++spin > DUMP_TIMEOUT_SPIN) {
                break;
            }
        }
        return value;
    }

    // 进度码取样（S/F/O 组相位，卡死取证用）
    __aicore__ inline int32_t DfxAt(uint32_t byteOff) const
    {
        return (dfxSeg_ != nullptr) ? ReadCnt(dfxSeg_, byteOff) : -1;
    }

    __aicore__ inline void WriteNbi(uint64_t handle, GM_ADDR dst, GM_ADDR src, uint64_t len)
    {
        int32_t ret = comm_->WriteNbi(handle, dst, src, len);
        ascendc_assert(ret == 0, "WriteNbi failed, ret=%d, rankId=%u, aivId=%u", ret, rankId_, aivId_);
    }

    __aicore__ inline void Drain(uint64_t handle)
    {
        int32_t ret = comm_->Drain(handle);
        ascendc_assert(ret == 0, "Drain failed, ret=%d, rankId=%u, aivId=%u", ret, rankId_, aivId_);
    }

    // 数据 + 计数 notify（失败 3 次重试）
    __aicore__ inline void WriteWithNotify(uint64_t handle, GM_ADDR dst, GM_ADDR src, uint64_t len, GM_ADDR notifyAddr,
                                           uint64_t notifyVal)
    {
        int32_t ret = comm_->WriteWithNotifyNbi(handle, dst, src, len, notifyAddr, notifyVal);
        if (ret != 0) {
            for (uint32_t i = 0; i < WRITE_RETRY_COUNT; i++) {
                ret = comm_->WriteWithNotifyNbi(handle, dst, src, len, notifyAddr, notifyVal);
                if (ret == 0) {
                    break;
                }
            }
            ascendc_assert(ret == 0, "WriteWithNotifyNbi failed after %u retries, ret=%d, rankId=%u, aivId=%u",
                           WRITE_RETRY_COUNT, ret, rankId_, aivId_);
        }
    }

    // 32B 远程计数写（done 用）：值经 scratch 暂存 → WriteNbi
    __aicore__ inline void WriteCounter32(uint64_t handle, GM_ADDR dst, int32_t value)
    {
        statusUb_.SetValue(0, value);
        EngramWinSyncFunc<AscendC::HardEvent::S_MTE3>(*pipe_);
        AscendC::GlobalTensor<int32_t> scratchGM;
        scratchGM.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(scratch_));
        AscendC::DataCopyParams params = {1U, static_cast<uint16_t>(UB_ALIGN), 0U, 0U};
        AscendC::DataCopyPad(scratchGM, statusUb_, params);
        EngramWinSyncFunc<AscendC::HardEvent::MTE3_S>(*pipe_);
        WriteNbi(handle, dst, scratch_, UB_ALIGN);
    }

    // 哨兵 0xDEADBEEF：与对端 read 计数垃圾值 0xFFFFFFFF 区分（合法计数恒为小正值）
    static constexpr int32_t CREDIT_SENTINEL = -559038737; // 0xDEADBEEF
    static constexpr int32_t CREDIT_CNT_GARBAGE = -1;      // 对端计数器垃圾特征值（0xFFFFFFFF）
    static constexpr uint32_t WRITE_RETRY_COUNT = 3U;
    static constexpr uint64_t DUMP_TIMEOUT_SPIN = 100000ULL; // 取证补读自旋上限

    AscendC::Hcomm<AscendC::COMM_PROTOCOL_UBC_CTP>* comm_{nullptr};
    GM_ADDR scratch_{0};
    AscendC::TPipe* pipe_{nullptr};
    AscendC::LocalTensor<int32_t> statusUb_;
    WinLayout w_{};
    uint32_t aivId_{0};
    uint32_t rankId_{0};
    uint32_t writeCnt_{0};     // 本端已发行数（累计，环形取模定位）
    int32_t remoteReadCnt_{0}; // 对端已读行数（信用）
    bool creditInFlight_{false};
    GM_ADDR remoteArea_{nullptr}; // 最近一次信用读的对端收件区（卡死取证补读用）
    uint64_t lastHandle_{0};      // 最近一次信用读的通道句柄
    GM_ADDR dfxSeg_{nullptr};     // 进度码区（可选；断言时取样各组相位）
};

// ConsumeFn: void(GM_ADDR blockGM, uint32_t rows)，块内行连续、含 32B 头；
// stallSeg/stallCores（可选）：S 核失败现场快照区，超时时随进度码一并打出
template <typename ConsumeFn>
__aicore__ inline void RecvLoop(GM_ADDR area, const WinLayout& w, AscendC::TPipe& pipe,
                                AscendC::LocalTensor<int32_t> statusUb, ConsumeFn&& consume, GM_ADDR dfxSeg = nullptr,
                                GM_ADDR stallSeg = nullptr, uint32_t stallCores = 0U)
{
    uint32_t localRead = 0;   // 已读行数（累计）
    uint64_t t0 = WinNowUs(); // 真墙钟 60s
    for (;;) {
        int32_t wr = ReadCnt(area, CNT_WRITE_OFF);
        if (wr > static_cast<int32_t>(localRead)) {
            uint32_t n = static_cast<uint32_t>(wr) - localRead;
            uint32_t t0r = localRead % w.numRows; // 跨环尾拆两段
            uint32_t fit = w.numRows - t0r;
            if (n <= fit) {
                consume(RowSlot(area, w, t0r), n);
            } else {
                consume(RowSlot(area, w, t0r), fit);
                consume(RowSlot(area, w, 0), n - fit);
            }
            localRead += n;
            WriteLocalReadCnt(area, localRead, pipe, statusUb); // 回写 read 计数（信用）
        }
        if (AreaFinished(area, static_cast<int32_t>(localRead))) {
            break;
        }
        if (WinNowUs() - t0 > WIN_TIMEOUT_US) {
            int32_t done = ReadCnt(area, CNT_DONE_OFF);
            // DFX：超时现场——F 进度 + 首个异常 S 核快照（快照位域见文件头 DFX 布局注释）
            int32_t fp0 = (dfxSeg != nullptr) ? ReadCnt(dfxSeg, DFX_F_PROG0_OFF) : -1;
            int32_t fp1 = (dfxSeg != nullptr) ? ReadCnt(dfxSeg, DFX_F_PROG1_OFF) : -1;
            int32_t stallCnt = 0;
            int32_t badC = -1;
            int32_t badW0 = 0;
            int32_t badW1 = -1;
            int32_t badW2 = -1;
            int32_t badW3 = -1;
            if (stallSeg != nullptr && stallCores > 0U) {
                uint32_t lim = (stallCores > DFX_STALL_SAMPLE_MAX) ? DFX_STALL_SAMPLE_MAX : stallCores;
                for (uint32_t c = 0; c < lim; ++c) {
                    int32_t w0 = ReadCnt(stallSeg, (DFX_SCALAR_SLOTS + c * DFX_SNAP_WORDS) * 4U);
                    if (w0 == 0) {
                        continue; // 无快照 = 该核已完结（或从未 idle）
                    }
                    stallCnt++;
                    bool bad = (((w0 & (1 << DFX_W0_OWN_DONE_BIT)) == 0) || ((w0 & (1 << DFX_W0_ALL_EOF_BIT)) == 0) ||
                                ((w0 & (1 << DFX_W0_DRAIN_EQ_BIT)) == 0));
                    if (bad && badC < 0) {
                        badC = static_cast<int32_t>(c);
                        badW0 = w0;
                        badW1 = ReadCnt(stallSeg, (DFX_SCALAR_SLOTS + 1U + c * DFX_SNAP_WORDS) * 4U);
                        badW2 = ReadCnt(stallSeg, (DFX_SCALAR_SLOTS + 2U + c * DFX_SNAP_WORDS) * 4U);
                        badW3 = ReadCnt(stallSeg, (DFX_SCALAR_SLOTS + 3U + c * DFX_SNAP_WORDS) * 4U);
                    }
                }
            }
            ascendc_assert(false,
                           "recv timeout: area=%llu, write=%d, read=%u, done=%d, fProg=%d/%d, "
                           "stallCnt=%d, bad=[c=%d w0=%d f=%d st=%d dr=%d eofV=%d]",
                           static_cast<uint64_t>(reinterpret_cast<uintptr_t>(area)), wr, localRead, done, fp0, fp1,
                           stallCnt, badC, badW0, badW1, badW2, badW3 & 0xFFFF, badW3 >> 16);
        }
    }
}

/// @brief GM->UB->GM 乒乓拷贝（64K 双缓冲、延迟一拍流水、块长 <= 65504）
__aicore__ inline void WinLocalCopy(GM_ADDR dst, GM_ADDR src, uint64_t len, AscendC::TPipe* pipe,
                                    AscendC::LocalTensor<uint8_t> ping, AscendC::LocalTensor<uint8_t> pong,
                                    int32_t* evtMte2ToMte3, int32_t* evtMte3ToMte2)
{
    constexpr uint32_t kMaxCopyBytes = 65504U;
    if (len == 0) {
        return;
    }
    AscendC::GlobalTensor<uint8_t> srcGm;
    AscendC::GlobalTensor<uint8_t> dstGm;
    srcGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(src));
    dstGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(dst));
    AscendC::DataCopyPadExtParams<uint8_t> pad{false, 0, 0, 0};

    uint64_t off = 0;
    uint32_t tileIdx = 0;
    uint64_t pendingOff = 0;
    uint32_t pendingLen = 0;
    uint32_t pendingBufIdx = 0;
    bool hasPendingMte3 = false;

    // 延迟一拍流水：本轮 MTE2 读新块，同时 MTE3 写出上一块(pending)
    while (off < len) {
        uint32_t thisLen = (len - off > kMaxCopyBytes) ? kMaxCopyBytes : static_cast<uint32_t>(len - off);
        uint32_t bufIdx = tileIdx % 2U;
        AscendC::LocalTensor<uint8_t>& buf = (bufIdx == 0U) ? ping : pong;

        if (tileIdx >= 2U) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(evtMte3ToMte2[bufIdx]);
        }
        AscendC::DataCopyExtParams mte2Params{1U, thisLen, 0U, 0U, 0U};
        AscendC::DataCopyPad(buf, srcGm[off], mte2Params, pad);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(evtMte2ToMte3[bufIdx]);

        if (hasPendingMte3) {
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(evtMte2ToMte3[pendingBufIdx]);
            AscendC::LocalTensor<uint8_t>& pendBuf = (pendingBufIdx == 0U) ? ping : pong;
            AscendC::DataCopyExtParams mte3Params{1U, pendingLen, 0U, 0U, 0U};
            AscendC::DataCopyPad(dstGm[pendingOff], pendBuf, mte3Params);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(evtMte3ToMte2[pendingBufIdx]);
        }
        pendingOff = off;
        pendingLen = thisLen;
        pendingBufIdx = bufIdx;
        hasPendingMte3 = true;
        off += thisLen;
        tileIdx++;
    }
    if (hasPendingMte3) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(evtMte2ToMte3[pendingBufIdx]);
        AscendC::LocalTensor<uint8_t>& pendBuf = (pendingBufIdx == 0U) ? ping : pong;
        AscendC::DataCopyExtParams mte3Params{1U, pendingLen, 0U, 0U, 0U};
        AscendC::DataCopyPad(dstGm[pendingOff], pendBuf, mte3Params);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(evtMte3ToMte2[pendingBufIdx]);
    }
    uint32_t last = (tileIdx - 1U) % 2U;
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(evtMte3ToMte2[last]);
    if (tileIdx >= 2U) {
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(evtMte3ToMte2[last ^ 1U]);
    }
    (void)pipe;
}

} // namespace Mc2Kernel

#endif // ENGRAM_FETCH_GRAD_WIN_H
