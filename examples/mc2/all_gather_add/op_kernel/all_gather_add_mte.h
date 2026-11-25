/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file all_gather_add_mte.h
 * \brief
 */
#ifndef ALL_GATHER_ADD_MTE_H
#define ALL_GATHER_ADD_MTE_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "all_gather_add_tiling.h"
#include "c_api/cache_ctrl/cache_ctrl.h"
#include "c_api/asc_simd.h"

//      A5 下 HcclOpParam 映射为 HcclA5OpResParam（固定 ABI 结构），kernel 必须按此布局解析。
constexpr uint32_t HCCL_MTE_MAX_RANK_NUM = 64;
struct HcclA5OpResParam {
    uint64_t workSpace;
    uint64_t workSpaceSize;
    uint32_t rankId;  // 当前卡 rankId
    uint32_t rankDim; // 总卡数
    uint64_t winSize;
    uint64_t windowsIn[HCCL_MTE_MAX_RANK_NUM];  // 完整 window 基址（状态区在 base，数据区在 +1MB）
    uint64_t windowsOut[HCCL_MTE_MAX_RANK_NUM]; // 双缓冲另一半（半区 1，winBufferFlags==1 时使用）
    uint64_t xnAddr;
    uint64_t ckeAddr;
    uint64_t msAddr;
    uint64_t msSize;
};

namespace AscendC {

// A5 核内同步：FetchEventID 轮转事件，避免固定 event id 冲突。
template <HardEvent event>
__aicore__ inline void SyncFunc()
{
    TEventID eventID = GetTPipePtr()->FetchEventID(event);
    SetFlag<event>(eventID);
    WaitFlag<event>(eventID);
}

constexpr int32_t ADD_BUFFER_NUM = 2; // 计算队列缓冲份数（串行复用，非流水线双缓冲）
constexpr uint64_t MTE_STATE_WIN_SIZE = 1024UL * 1024UL; // A5 状态区固定 1MB，数据区紧随其后
constexpr uint32_t SPLIT_BLOCK_HALF = 256;               // 一个 512B 块 = 256 half
constexpr uint32_t SPLIT_BLOCK_DATA_HALF = 240;          // 每块数据 = 240 half(480B)
constexpr uint32_t FLAG_HALF = 16;                       // 每块 flag 槽 = 16 half(32B)
constexpr uint32_t FLAG_U32 = FLAG_HALF * sizeof(half) / sizeof(uint32_t); // 32B = 8 个 uint32
constexpr uint32_t MTE_FLAG_READY = 0x1U;                                  // 就绪标志值
constexpr uint32_t MTE_FLAG_CLEAR = 0x0U;                                  // 复位标志值
constexpr int32_t SYNC_FLAG_NUM = 8;                 // 发送循环里事件 flag 的轮转数（对应 8 个 UB 槽）
constexpr uint64_t WIN_FLAG_OFFSET = 924UL * 1024UL; // 状态区 0/1 分区标志位偏移（参考 quant_all_reduce 标志位区）
constexpr uint32_t MTE_TIMEOUT_ERR_CODE =
    0xDEAD0001U; // WaitPeers 超时唯一错误码，trap 前写进 gatherOut 首个 uint32 供 host 确认
constexpr uint32_t WAIT_PEERS_SPIN_LIMIT = 500000U; // 等待对端 READY 的自旋上限，超时走 trap 失败路径

template <typename T>
class AllGatherAddMte {
public:
    __aicore__ inline AllGatherAddMte() {}
    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR gatherGM, AllGatherAddTilingData* tiling,
                                TPipe* pipe);
    __aicore__ inline void Process();

private:
    // 各 rank 窗口的数据区基址：0/1 双缓冲，半区 0 = windowsIn+1MB，半区 1 = windowsOut。
    __aicore__ inline GM_ADDR WindDataAddr(uint32_t rankId)
    {
        if (winBufferFlags_ == 0U) {
            return (GM_ADDR)ctx_->windowsIn[rankId] + MTE_STATE_WIN_SIZE;
        }
        return (GM_ADDR)ctx_->windowsOut[rankId];
    }
    // 数据区里某个 rank 的 slot 起始偏移（单位 half）
    __aicore__ inline uint64_t SlotDataOffset(uint32_t rankId)
    {
        return static_cast<uint64_t>(rankId) * slotStride_;
    }
    __aicore__ inline void SendToPeers();
    __aicore__ inline bool WaitPeers();
    __aicore__ inline void GatherAndAdd();
    __aicore__ inline void ToggleWinFlag();

    AllGatherAddTilingData* tiling_{nullptr};
    __gm__ HcclA5OpResParam* ctx_{nullptr}; // 框架 HcclAllocComResourceByTiling 建好的 MTE 窗口上下文
    uint32_t myRank_{0};
    uint32_t rankDim_{0};           // all_gather 的总 rank 数
    uint32_t aElemsPerRank_{0};     // 每个 rank 的 a 元素数
    uint32_t numBlocks_{0};         // 发送分块数 = aElemsPerRank / 240
    uint32_t slotStride_{0};        // 数据区每个 rank slot 的步长（half）= numBlocks_*256
    uint32_t blockElemNumTotal_{0}; // 每个核要处理的 gather_out 元素数 = totalElemNum/blockDim
    uint32_t addTileElemNum_{0};    // 每个计算 tile 的元素数（封顶后）
    uint32_t tileNum_{0};           // 计算 tile 数（含尾块）
    uint32_t blockIdx_{0};          // 当前核 id（AIV core id）
    uint32_t blocksPerTile_{0};     // 每个 tile 读回的数据块数
    uint32_t winBufferFlags_{0};    // 0/1 分区标志位（双缓冲当前半区）

    GM_ADDR aGM_{0};
    GM_ADDR bGM_{0};
    GM_ADDR cGM_{0};
    GM_ADDR gatherGM_{0};

    TPipe* pipe_{nullptr};
    TBuf<> sendBuf_;                                   // 清 flag / 轮询 flag 用的小临时 UB
    TQue<QuePosition::VECIN, SYNC_FLAG_NUM> sendQ_;    // 发送循环：MTE2 读入队列，EnQue/DeQue 同步
    TQue<QuePosition::VECIN, ADD_BUFFER_NUM> gatherQ_; // gather 数据队列
    TQue<QuePosition::VECIN, ADD_BUFFER_NUM> bQ_;      // b 数据队列
    TQue<QuePosition::VECOUT, ADD_BUFFER_NUM> cQ_;     // c 结果队列
};

template <typename T>
__aicore__ inline void AllGatherAddMte<T>::Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR gatherGM,
                                                AllGatherAddTilingData* tiling, TPipe* pipe)
{
    aGM_ = aGM;
    bGM_ = bGM;
    cGM_ = cGM;
    gatherGM_ = gatherGM;

    tiling_ = tiling;
    pipe_ = pipe;
    blockIdx_ = GetBlockIdx();

    // 走框架 GetHcclContext 读 MTE 窗口（HcclAllocComResourceByTiling 建好的 HcclA5OpResParam）。
    ctx_ = (__gm__ HcclA5OpResParam*)GetHcclContext<HCCL_GROUP_ID_0>();
    myRank_ = ctx_->rankId;
    rankDim_ = ctx_->rankDim; // 总卡数（框架保证 1..64，对应 windowsIn/windowsOut 数组上界）

    aElemsPerRank_ = tiling_->totalElemNum / rankDim_;   // 每 rank 的 a 元素数 = totalElemNum/rankDim
    numBlocks_ = aElemsPerRank_ / SPLIT_BLOCK_DATA_HALF; // 发送分块数 = aElemsPerRank/240
    slotStride_ = numBlocks_ * SPLIT_BLOCK_HALF;         // 数据区每个 rank slot 步长（half）= numBlocks*256
    blockElemNumTotal_ =
        tiling_->blockElemNum * tiling_->commTurn; // = totalElemNum/blockDim（MTE 单轮发送，commTurn 乘回抵消）
    addTileElemNum_ = tiling_->addTileElemNum;     // 封顶后的 tile（tiling 下发）
    tileNum_ = tiling_->tileNum;                   // tile 数（含尾块）
    blocksPerTile_ = addTileElemNum_ / SPLIT_BLOCK_DATA_HALF; // 每 tile 的 gather 块数

    // sendBuf_：清 flag/轮询 flag 用的小临时区；sendQ_：发送循环 8 个 512B 槽（队列同步）
    pipe_->InitBuffer(sendBuf_, SYNC_FLAG_NUM * SPLIT_BLOCK_HALF * sizeof(T));
    pipe_->InitBuffer(sendQ_, SYNC_FLAG_NUM, SPLIT_BLOCK_HALF * sizeof(T));
    pipe_->InitBuffer(gatherQ_, ADD_BUFFER_NUM, addTileElemNum_ * sizeof(T));
    pipe_->InitBuffer(bQ_, ADD_BUFFER_NUM, addTileElemNum_ * sizeof(T));
    pipe_->InitBuffer(cQ_, ADD_BUFFER_NUM, addTileElemNum_ * sizeof(T));
}

template <typename T>
__aicore__ inline void AllGatherAddMte<T>::SendToPeers()
{
    if (blockIdx_ != 0) { // 只有 core0 负责发送
        return;
    }
    GlobalTensor<T> aGm;
    aGm.SetGlobalBuffer((__gm__ T*)aGM_);
    aGm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);

    {
        LocalTensor<uint32_t> clearFlag = sendBuf_.Get<uint32_t>();
        for (uint32_t i = 0; i < FLAG_U32; ++i) {
            clearFlag(i) = MTE_FLAG_CLEAR; // 整个 32B flag 槽清零（WaitPeers 只读首 uint32，其余避免残留垃圾）
        }
        SyncFunc<HardEvent::S_MTE3>();
        DataCopyExtParams flagWParams{1U, static_cast<uint32_t>(FLAG_U32 * sizeof(uint32_t)), 0U, 0U, 0U};
        for (uint32_t r = 0; r < rankDim_; ++r) {
            for (uint32_t b = 0; b < numBlocks_; ++b) {
                GlobalTensor<uint32_t> flagDst;
                flagDst.SetGlobalBuffer(
                    (__gm__ uint32_t*)(WindDataAddr(r) +
                                       (SlotDataOffset(myRank_) + static_cast<uint64_t>(b) * SPLIT_BLOCK_HALF +
                                        SPLIT_BLOCK_DATA_HALF) *
                                           sizeof(T)));
                flagDst.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
                DataCopyPad(flagDst, clearFlag, flagWParams); // MTE3 写 32B 清空 flag 槽
            }
        }
        SyncFunc<HardEvent::MTE3_V>();
    }

    // [阶段1] 分块发送：MTE2 读 a → 填内联 flag READY → MTE3 写所有 rank 窗口（数据+flag 一次写完）
    for (uint32_t b = 0; b < numBlocks_; ++b) {
        // ── [MTE2 读] GM(a) → UB(in) ──
        LocalTensor<T> in = sendQ_.AllocTensor<T>();                         // 从 sendQ_ 取一个 512B UB 槽
        DataCopy(in, aGm[b * SPLIT_BLOCK_DATA_HALF], SPLIT_BLOCK_DATA_HALF); // MTE2 异步读 240 half
        sendQ_.EnQue(in);                                                    // 入队：登记本槽 MTE2 读已提交

        LocalTensor<T> inReady = sendQ_.DeQue<T>(); // 出队：等本槽 MTE2 读完成
        // 完成 flag：显式 MTE2→Scalar，补齐 VECIN 队列对 MTE2→MTE3 依赖的缺失
        SyncFunc<HardEvent::MTE2_S>();

        // ── [标量写] 内联 flag 槽（第一个 uint32）置 READY ──
        LocalTensor<uint32_t> flagSlot = inReady[SPLIT_BLOCK_DATA_HALF].template ReinterpretCast<uint32_t>();
        flagSlot(0) = MTE_FLAG_READY;
        SyncFunc<HardEvent::S_MTE3>();

        // ── [MTE3 写] UB(inReady) → GM(每个 rank 窗口) ──
        for (uint32_t r = 0; r < rankDim_; ++r) { // 写满所有 rank（含本卡 loopback）
            GlobalTensor<T> dst;
            dst.SetGlobalBuffer(
                (__gm__ T*)(WindDataAddr(r) +
                            (SlotDataOffset(myRank_) + static_cast<uint64_t>(b) * SPLIT_BLOCK_HALF) * sizeof(T)));
            dst.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
            DataCopy(dst, inReady, SPLIT_BLOCK_HALF); // MTE3 写 256 half = 512B（480B 数据 + 32B 内联 flag）
        }
        // 完成 flag：MTE3→Vector，保证 MTE3 已读完 inReady，UB 槽才能释放复用
        SyncFunc<HardEvent::MTE3_V>();
        sendQ_.FreeTensor(inReady);
    }
}

template <typename T>
__aicore__ inline bool AllGatherAddMte<T>::WaitPeers()
{
    if (blockIdx_ != 0) {
        return true; // 非 block0 无等待职责
    }
    LocalTensor<uint32_t> flag = sendBuf_.Get<uint32_t>();
    DataCopyExtParams flagRParams{1U, static_cast<uint32_t>(FLAG_U32 * sizeof(uint32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<uint32_t> flagRPadParams{false, 0U, 0U, 0U};

    uint32_t spin = 0;
    uint32_t lastSrc = 0;
    uint32_t lastBlock = 0;
    uint32_t lastFlagVal = 0;
    bool allReady = false;
    while (!allReady && spin < WAIT_PEERS_SPIN_LIMIT) {
        allReady = true;
        for (uint32_t src = 0; src < rankDim_; ++src) { // 0=自己数据(slot0)，1=对端数据(slot1)
            for (uint32_t b = 0; b < numBlocks_; ++b) {
                // src slot 里第 b 块的内联 flag 地址 = 数据区 + src 的 slot + 块内 flag 槽偏移
                GM_ADDR flagAddr =
                    WindDataAddr(myRank_) +
                    (SlotDataOffset(src) + static_cast<uint64_t>(b) * SPLIT_BLOCK_HALF + SPLIT_BLOCK_DATA_HALF) *
                        sizeof(T);
                GlobalTensor<uint32_t> flagGm;
                flagGm.SetGlobalBuffer((__gm__ uint32_t*)flagAddr);
                DataCopyPad(flag, flagGm, flagRParams, flagRPadParams); // MTE2 读 32B 内联 flag
                SyncFunc<HardEvent::MTE2_S>();
                lastFlagVal = flag.GetValue(0);
                if (lastFlagVal != MTE_FLAG_READY) {
                    allReady = false;
                    lastSrc = src;
                    lastBlock = b;
                    break;
                }
            }
            if (!allReady) {
                break;
            }
        }
        ++spin;
    }

    if (!allReady) {
        // 超时：等待结束 ≠ 通信完成，必须显式失败，不能把未就绪窗口当有效数据读。
        AscendC::printf("[AllGatherAdd][WaitPeers] timeout: rank=%u src=%u block=%u spin=%u flag=0x%x\n", myRank_,
                        lastSrc, lastBlock, spin, lastFlagVal);
        return false;
    }
    return true;
}

template <typename T>
__aicore__ inline void AllGatherAddMte<T>::GatherAndAdd()
{
    // 本核要处理的全局偏移、源 rank、块起点
    uint32_t totalOffset = blockIdx_ * blockElemNumTotal_;
    uint32_t srcRank = totalOffset / aElemsPerRank_;
    uint32_t rankOffset = totalOffset - srcRank * aElemsPerRank_;
    uint32_t blockStart = rankOffset / SPLIT_BLOCK_DATA_HALF;

    // 从本卡窗口的数据区 + srcRank 的 slot + blockStart 处读起
    GM_ADDR gatherBase = WindDataAddr(myRank_) +
                         (SlotDataOffset(srcRank) + static_cast<uint64_t>(blockStart) * SPLIT_BLOCK_HALF) * sizeof(T);

    GlobalTensor<T> gatherGm;
    gatherGm.SetGlobalBuffer((__gm__ T*)gatherBase);
    GlobalTensor<T> bGm;
    bGm.SetGlobalBuffer((__gm__ T*)bGM_ + totalOffset);
    bGm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
    GlobalTensor<T> cGm;
    cGm.SetGlobalBuffer((__gm__ T*)cGM_ + totalOffset);
    GlobalTensor<T> gatherOutGm;
    gatherOutGm.SetGlobalBuffer((__gm__ T*)gatherGM_ + totalOffset, blockElemNumTotal_);

    // Route B：gather 读分块连续 DataCopy（走 L2）

    // 分块循环：每 tile 处理 addTileElemNum_ half（尾块更少），复用 gatherQ_/bQ_/cQ_ 双缓冲
    for (uint32_t j = 0; j < tileNum_; j++) {
        uint32_t tileElems = addTileElemNum_;
        uint32_t remain = blockElemNumTotal_ - j * addTileElemNum_;
        if (remain < tileElems) {
            tileElems = remain; // 尾块
        }
        uint32_t tileBlocks = tileElems / SPLIT_BLOCK_DATA_HALF;

        // ── [MTE2 读] 窗口 gather → UB(gLocal)、GM(b) → UB(bLocal) ──
        LocalTensor<T> gLocal = gatherQ_.AllocTensor<T>();
        LocalTensor<T> bLocal = bQ_.AllocTensor<T>();
        // Route B：分块连续读（走 L2），每块 240 half，块 k 数据起点 = tile 首 half + k*256
        for (uint32_t k = 0; k < tileBlocks; ++k) {
            DataCopy(gLocal[k * SPLIT_BLOCK_DATA_HALF],
                     gatherGm[j * blocksPerTile_ * SPLIT_BLOCK_HALF + k * SPLIT_BLOCK_HALF], SPLIT_BLOCK_DATA_HALF);
        }
        DataCopy(bLocal, bGm[j * addTileElemNum_], tileElems); // MTE2 读 b
        gatherQ_.EnQue(gLocal);                                // 完成 flag 入队：gather 数据已就绪
        bQ_.EnQue(bLocal);                                     // 完成 flag 入队：b 数据已就绪

        LocalTensor<T> gDeq = gatherQ_.DeQue<T>(); // 等 gather 读完成
        LocalTensor<T> bDeq = bQ_.DeQue<T>();      // 等 b 读完成
        SyncFunc<HardEvent::MTE2_S>();             // 等所有 MTE2 读完成

        // ── [MTE3 写] UB(gDeq) → GM(gatherOut) ──
        DataCopy(gatherOutGm[j * addTileElemNum_], gDeq, tileElems); // MTE3 读 gDeq 写 gatherOut

        // ── [V 计算] c = gather + b ──
        LocalTensor<T> cLocal = cQ_.AllocTensor<T>();
        Add(cLocal, gDeq, bDeq, tileElems); // Vector 读 gDeq/bDeq
        cQ_.EnQue<T>(cLocal);

        SyncFunc<HardEvent::MTE3_V>(); // 完成 flag：MTE3（写 gatherOut）已读完 gDeq
        SyncFunc<HardEvent::V_MTE3>(); // 完成 flag：Vector（Add）已读完 gDeq/bDeq
        gatherQ_.FreeTensor(gDeq);
        bQ_.FreeTensor(bDeq);

        // ── [MTE3 写] UB(cDeq) → GM(c) ──
        LocalTensor<T> cDeq = cQ_.DeQue<T>();
        DataCopy(cGm[j * addTileElemNum_], cDeq, tileElems); // MTE3 读 cDeq 写 c
        SyncFunc<HardEvent::MTE3_V>();                       // 完成 flag：MTE3 已读完 cDeq，槽可释放
        cQ_.FreeTensor(cDeq);
    }

    // 冲刷整个 data cache 到 HBM：gatherOut 和 c 都由 MTE3 写入 L2，host 侧读 HBM，需冲刷后才可见。
    DataCacheCleanAndInvalid<T, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(gatherOutGm);
}

template <typename T>
__aicore__ inline void AllGatherAddMte<T>::ToggleWinFlag()
{
    // 0/1 分区标志位持久存在状态区里，每轮调用翻转一次，实现数据区双缓冲：
    // 半区 0 = windowsIn+1MB，半区 1 = windowsOut。翻转后所有核读回本轮使用的半区。
    // 这样第 N 轮写半区 X、第 N+1 轮写半区 1-X，快 rank 覆盖新半区时不会踩到慢 rank 正在读的旧半区。
    GM_ADDR flagAddr = (GM_ADDR)ctx_->windowsIn[myRank_] + WIN_FLAG_OFFSET;
    GlobalTensor<uint32_t> flagGm;
    flagGm.SetGlobalBuffer((__gm__ uint32_t*)flagAddr);
    flagGm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE); // 直走 HBM，保证跨调用持久、跨核可见

    if (blockIdx_ == 0) {
        LocalTensor<uint32_t> flagLocal = sendBuf_.Get<uint32_t>();
        DataCopy(flagLocal, flagGm, FLAG_U32); // MTE2 读 32B（含旧 flag，其余字节保留）
        SyncFunc<HardEvent::MTE2_S>();
        uint32_t curFlag = flagLocal.GetValue(0);
        // 归一化 + 翻转：非 1 一律翻成 1，1 翻成 0。状态区首轮若是脏值（非 0/1）也能正确进入 0↔1 交替，
        // 避免 1-curFlag 在脏值上循环出非 0/1 的大数、导致永远选同一半区（双缓冲失效）。
        // 跨卡一致性依赖两卡首轮 flag 初值一致（框架零初始化）；「非 1 都映射到 1」保证大多数脏值下两卡同半区，
        // 仅当一卡脏值恰为 1、另一卡非 1 才会错位（概率 ~1/2^32，可忽略）。
        flagLocal.SetValue(0, (curFlag == 1U) ? 0U : 1U);
        SyncFunc<HardEvent::S_MTE3>();
        DataCopy(flagGm, flagLocal, FLAG_U32); // MTE3 写回 32B
        SyncFunc<HardEvent::MTE3_V>();
    }

    SyncAll<true>(); // 等 block0 写回，其它核可见

    LocalTensor<uint32_t> flagLocal = sendBuf_.Get<uint32_t>();
    DataCopy(flagLocal, flagGm, FLAG_U32);
    SyncFunc<HardEvent::MTE2_S>();
    winBufferFlags_ = flagLocal.GetValue(0);
}

template <typename T>
__aicore__ inline void AllGatherAddMte<T>::Process()
{
    ToggleWinFlag();           // 0) 翻转 0/1 分区，选本轮数据半区（双缓冲，避免覆盖竞态）
    SendToPeers();             // 1) block0 发送本 rank 的 a 到本轮半区
    SyncAll<true>();           // 2) 全核 barrier
    bool waitOk = WaitPeers(); // 3) block0 等对端 ready flag；超时返回 false
    if (!waitOk) {
        // 通信超时：先写唯一错误码到 gatherOut 首个 uint32，再 trap 中止整个 kernel。
        // host 侧 sync 失败后读回确认「是 WaitPeers 超时」；绝不继续 GatherAndAdd 读未就绪窗口。
        if (blockIdx_ == 0) {
            GlobalTensor<uint32_t> errGm;
            errGm.SetGlobalBuffer((__gm__ uint32_t*)gatherGM_);
            errGm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE); // 直写 HBM，trap 后 host 仍可读
            LocalTensor<uint32_t> errLocal = sendBuf_.Get<uint32_t>();
            errLocal.SetValue(0, MTE_TIMEOUT_ERR_CODE);
            SyncFunc<HardEvent::S_MTE3>();
            DataCopy(errGm, errLocal, FLAG_U32); // 32B 写，与 flag 写保持一致；host 只读首个 uint32
            SyncFunc<HardEvent::MTE3_V>();
        }
        AscendC::Trap();
    }
    SyncAll<true>(); // 4) 全核 barrier
    GatherAndAdd();  // 5) 所有核读本轮半区、算 gather_out 和 c
}

} // namespace AscendC
#endif
