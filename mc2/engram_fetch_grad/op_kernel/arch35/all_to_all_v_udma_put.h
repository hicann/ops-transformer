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
 * \file all_to_all_v_udma_put.h
 * \brief AllToAllv(变长) PUT 模式实现 - 基于 hcomm WriteWithNotifyNbi
 */

#ifndef ALL_TO_ALL_V_UDMA_PUT_H
#define ALL_TO_ALL_V_UDMA_PUT_H

#include "kernel_operator.h"
#include "adv_api/hccl/hccl.h"

namespace Mc2Kernel {

using namespace AscendC;

struct AllToAllvConfig {
    uint32_t rankId{0};
    uint32_t rankSize{0};
    uint32_t channelsPerRank{1}; // 到每个 peer 预建的通信通道数
    uint64_t winSize{0};         // 通信窗口总容量(字节)
    uint32_t numSlots{8};        // 每 rank 窗口槽数(发送窗口深度单位)
    uint64_t rowBytes{0};        // 单行字节数

    GM_ADDR srcGM{0};        // 发送源(本 rank 已就绪的变长数据, 按 sdispl 索引)
    GM_ADDR dstGM{0};        // 接收目的(按 rdispl 索引落盘)
    GM_ADDR sendCountsGM{0}; // [rankSize] int32: 发往各 peer 的行数
    GM_ADDR recvCountsGM{0}; // [rankSize] int32: 从各 peer 收到的行数
    GM_ADDR sdisplsGM{0};    // [rankSize] int64: 发送源内各 peer 段起始行号
    GM_ADDR rdisplsGM{0};    // [rankSize] int64: 接收目的内各 peer 段起始行号

    GM_ADDR creditScratchGM{0}; // credit 异步读暂存区

    // 窗口头保留字节数: [0, headReserveBytes) 为调用方的专用区(如跨kernel屏障槽),
    // a2a 的计数区/数据区从该偏移起布局。默认 0 = 占满全窗(与旧版布局一致)。
    // 同一窗口若被多个 kernel 复用, 各 kernel 的 headReserveBytes 必须一致。
    uint64_t headReserveBytes{0};
};

static constexpr uint32_t ALLTOALLV_COUNTER_SLOT_BYTES = 32U; // 计数器槽步长
static constexpr uint32_t ALLTOALLV_EQUAL_REGION_COUNT = 3U; // 窗口等宽计数区数(barrier标志/write计数/read credit)
static constexpr uint32_t ALLTOALLV_UB_ALIGN = 32U;          // UB/窗口 32B 对齐单位
static constexpr int32_t ALLTOALLV_CREDIT_SENTINEL = -1;
static constexpr uint32_t ALLTOALLV_MAX_PENDING = 8U; // 每 job 暂存未排空句柄的上限(攒满一批再统一 Drain)
static constexpr uint32_t ALLTOALLV_MAX_COPY_BYTES = 65504U; // 单次 DataCopy 块上限
static constexpr uint32_t ALLTOALLV_RETRY_COUNT = 3U;        // NBI 写失败重试

template <AscendC::HardEvent event>
__aicore__ inline void AllToAllvSyncFunc(AscendC::TPipe& pipe)
{
    int32_t eventID = static_cast<int32_t>(pipe.FetchEventID(event));
    AscendC::SetFlag<event>(eventID);
    AscendC::WaitFlag<event>(eventID);
}

class AllToAllvPut {
public:
    __aicore__ inline AllToAllvPut() = default;

    __aicore__ inline void Init(const AllToAllvConfig& config, __gm__ const uint64_t* channelHandles,
                                __gm__ const uint64_t* winAddrs, AscendC::Hcomm<COMM_PROTOCOL_UBC_CTP>& comm,
                                AscendC::LocalTensor<int32_t> statusUb, AscendC::LocalTensor<uint8_t> copyPing,
                                AscendC::LocalTensor<uint8_t> copyPong, AscendC::TPipe* pipe)
    {
        config_ = config;
        handles_ = channelHandles;
        winAddrs_ = winAddrs;
        comm_ = &comm;
        statusUb_ = statusUb; // credit/计数器读写 UB 视图
        copyPing_ = copyPing; // copyPing/copyPong 本地段拷贝的 UB 乒乓
        copyPong_ = copyPong;
        pipe_ = pipe;
        localWinBase_ = (GM_ADDR)winAddrs_[config_.rankId];
        pendingHandleCount_ = 0;
        creditReadInFlight_ = false;

        evtMte2ToMte3_[0] = static_cast<int32_t>(pipe_->FetchEventID(HardEvent::MTE2_MTE3));
        evtMte2ToMte3_[1] = static_cast<int32_t>(pipe_->FetchEventID(HardEvent::MTE2_MTE3));
        evtMte3ToMte2_[0] = static_cast<int32_t>(pipe_->FetchEventID(HardEvent::MTE3_MTE2));
        evtMte3ToMte2_[1] = static_cast<int32_t>(pipe_->FetchEventID(HardEvent::MTE3_MTE2));
    }

    __aicore__ inline void InitJobs(uint32_t jobBudget, uint32_t jobIndex)
    {
        // jobBudget a2a 总 job 数(= 参与 a2a 的核数)
        // jobIndex 本核在 a2a 域内的编号(发送 job 为前缀 [0, sendCnt), 接收为其后)
        jobIndex_ = jobIndex;
        uint32_t maxPipelinesPerRank =
            (config_.numSlots < config_.channelsPerRank) ? config_.numSlots : config_.channelsPerRank;
        // 每 rank 的 核份额 = 预算 / (R + (R-1))
        pipelinesPerRank_ = jobBudget / (2 * config_.rankSize - 1);
        // 控核场景需要，否则例如4send核8rank会出现sendperank为0
        if (pipelinesPerRank_ < 1U) {
            pipelinesPerRank_ = 1U;
        }
        if (pipelinesPerRank_ > maxPipelinesPerRank) {
            pipelinesPerRank_ = maxPipelinesPerRank;
        }
        // 至少需要2个任务，分给send一个，recv一个
        ascendc_assert(jobBudget >= 2U, "invalid jobBudget=%u (need >= 2: 1 send + 1 recv)", jobBudget);
        uint32_t sendJobs = jobBudget * config_.rankSize / (2 * config_.rankSize - 1);
        // 整除floor，故需要再判断下是否真的大于，否则限制
        if (sendJobs > pipelinesPerRank_ * config_.rankSize) {
            sendJobs = pipelinesPerRank_ * config_.rankSize; // job 多于流水线无意义
        }
        sendJobCnt_ = sendJobs;
        recvJobCnt_ = jobBudget - sendJobs;
        DeriveLayout(); // 布局推导
    }

    __aicore__ inline void ClearCounters(AscendC::LocalTensor<int32_t> zeroUb)
    {
        if (jobIndex_ != 0U) {
            return;
        }
        uint32_t cleanElements = config_.rankSize * pipelinesPerRank_ * ALLTOALLV_COUNTER_SLOT_BYTES / sizeof(int32_t);
        uint32_t totalCleanElements = 2U * cleanElements;
        Duplicate<int32_t>(zeroUb, 0, totalCleanElements);
        AllToAllvSyncFunc<HardEvent::V_MTE3>(*pipe_);
        GlobalTensor<int32_t> countersGM;
        countersGM.SetGlobalBuffer((__gm__ int32_t*)(localWinBase_ + tokenWriteOffset_));
        DataCopy(countersGM, zeroUb, totalCleanElements);
        AllToAllvSyncFunc<HardEvent::MTE3_S>(*pipe_);
    }

    __aicore__ inline void SendPhase()
    {
        pendingHandleCount_ = 0;
        if (jobIndex_ >= sendJobCnt_) {
            return; // 非 send job不参与发送
        }
        uint32_t totalSendWorkUnits = config_.rankSize * pipelinesPerRank_;
        // send核可能小于send任务数，需要取余方式
        for (uint32_t wIdx = jobIndex_ % totalSendWorkUnits; wIdx < totalSendWorkUnits; wIdx += sendJobCnt_) {
            uint32_t dstRank = wIdx % config_.rankSize;
            uint32_t pipelineIdx =
                wIdx / config_.rankSize; // 例如16个核，前8个负责0-7rank，后8个也负责0-7rank，只是不同的pipelineIdx

            GlobalTensor<int32_t> sSlotGM;
            sSlotGM.SetGlobalBuffer((__gm__ int32_t*)(config_.sendCountsGM + dstRank * ALLTOALLV_UB_ALIGN));
            DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(sSlotGM);
            int32_t totalSendCount = sSlotGM.GetValue(0);

            GlobalTensor<int64_t> sDisplSlotGM;
            sDisplSlotGM.SetGlobalBuffer((__gm__ int64_t*)(config_.sdisplsGM + dstRank * ALLTOALLV_UB_ALIGN));
            DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(sDisplSlotGM);
            int64_t sdispl = sDisplSlotGM.GetValue(0); // 前缀和

            // send前缀和+要send的count数
            int32_t baseShare = totalSendCount / static_cast<int32_t>(pipelinesPerRank_);
            int32_t mySendCount = baseShare;
            int64_t mySdispl = sdispl + static_cast<int64_t>(pipelineIdx) * baseShare;
            if (pipelineIdx == pipelinesPerRank_ - 1U) {
                mySendCount = totalSendCount - static_cast<int32_t>(pipelineIdx) * baseShare;
            }

            if (mySendCount <= 0) {
                continue;
            }
            // 目标卡为本卡走localcopy
            if (dstRank == config_.rankId) {
                GlobalTensor<int64_t> rDisplSlotGM;
                rDisplSlotGM.SetGlobalBuffer(
                    (__gm__ int64_t*)(config_.rdisplsGM + config_.rankId * ALLTOALLV_UB_ALIGN));
                DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(rDisplSlotGM);
                int64_t rdispl = rDisplSlotGM.GetValue(0);
                int64_t myRdispl = rdispl + static_cast<int64_t>(pipelineIdx) * baseShare;
                LocalCopy(config_.dstGM + static_cast<uint64_t>(myRdispl) * config_.rowBytes,
                          config_.srcGM + static_cast<uint64_t>(mySdispl) * config_.rowBytes,
                          static_cast<uint64_t>(mySendCount) * config_.rowBytes);
                continue;
            }
            SendUnitRemote(dstRank, pipelineIdx, mySendCount, mySdispl);
        }
    }

    __aicore__ inline void RecvPhase()
    {
        if (jobIndex_ < sendJobCnt_ || jobIndex_ >= sendJobCnt_ + recvJobCnt_) {
            return; // 非 recv job 不参与接收
        }
        uint32_t recvJobIdx = jobIndex_ - sendJobCnt_;
        uint32_t totalWorkUnits = (config_.rankSize - 1U) * pipelinesPerRank_;
        for (uint32_t wIdx = recvJobIdx; wIdx < totalWorkUnits; wIdx += recvJobCnt_) {
            uint32_t adjustedSrcRank = wIdx / pipelinesPerRank_;
            uint32_t srcRank = (adjustedSrcRank >= config_.rankId) ? (adjustedSrcRank + 1U) : adjustedSrcRank;
            uint32_t pipelineIdx = wIdx % pipelinesPerRank_;

            GlobalTensor<int32_t> rSlotGM;
            rSlotGM.SetGlobalBuffer((__gm__ int32_t*)(config_.recvCountsGM + srcRank * sizeof(int32_t)));
            DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(rSlotGM);
            int32_t recvCount = rSlotGM.GetValue(0);

            GlobalTensor<int64_t> rDisplSlotGM;
            rDisplSlotGM.SetGlobalBuffer((__gm__ int64_t*)(config_.rdisplsGM + srcRank * ALLTOALLV_UB_ALIGN));
            DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(rDisplSlotGM);
            int64_t rdispl = rDisplSlotGM.GetValue(0);

            int32_t baseShare = recvCount / static_cast<int32_t>(pipelinesPerRank_);
            uint32_t pipelineRecvCount =
                (pipelineIdx == pipelinesPerRank_ - 1U) ?
                    static_cast<uint32_t>(recvCount - static_cast<int32_t>(pipelineIdx) * baseShare) :
                    static_cast<uint32_t>(baseShare);
            if (pipelineRecvCount == 0U) {
                continue;
            }

            int64_t pipelineBase = rdispl + static_cast<int64_t>(pipelineIdx) * baseShare;
            uint32_t slotsPerSender = config_.numSlots / pipelinesPerRank_;
            uint64_t slotAreaBase =
                tokenDataOffset_ + static_cast<uint64_t>(srcRank) * config_.numSlots * tokenSlotSize_;

            uint32_t pipelineReceived = 0;
            uint32_t localReadCnt = 0;
            // 没收完就一直while收
            while (pipelineReceived < pipelineRecvCount) {
                int32_t remoteWriteCnt = ReadLocalCounter(tokenWriteOffset_, srcRank, pipelineIdx); // 对端写的槽数
                // 如果对端写的槽数不够我读，就while等待新槽到达
                while (remoteWriteCnt <= 0 || static_cast<uint32_t>(remoteWriteCnt) <= localReadCnt) {
                    remoteWriteCnt = ReadLocalCounter(tokenWriteOffset_, srcRank, pipelineIdx);
                }

                uint32_t availSlots = static_cast<uint32_t>(remoteWriteCnt) - localReadCnt;
                uint32_t remaining = pipelineRecvCount - pipelineReceived;
                uint32_t remainingChunks = (remaining + maxTokensPerSlot_ - 1U) / maxTokensPerSlot_;
                uint32_t localSlotIdx = localReadCnt % slotsPerSender;
                uint32_t maxBatchFromHere = slotsPerSender - localSlotIdx; // 从当前槽到本流水线槽组末尾的槽数
                // 算这一批能连续读几个槽batchSlots
                uint32_t batchSlots = availSlots;
                if (batchSlots > remainingChunks) {
                    batchSlots = remainingChunks;
                }
                if (batchSlots > maxBatchFromHere) {
                    batchSlots = maxBatchFromHere;
                }

                uint32_t slotsRead = 0;
                uint32_t tokensRead = 0;

                // 接受的slot数据无空隙时大批量拷贝, 否则逐槽（因为存在脏数据）
                bool noSlotSlack = (tokenSlotSize_ == static_cast<uint64_t>(maxTokensPerSlot_) * config_.rowBytes);
                if (noSlotSlack && batchSlots < remainingChunks) {
                    // 无间隙，且批量处理的槽数据没超过还剩余部分，可以一次性copy处理
                    uint32_t totalTokens = batchSlots * maxTokensPerSlot_;
                    uint32_t firstSegIdx = localReadCnt % slotsPerSender;
                    uint32_t firstSlotIdx = pipelineIdx * slotsPerSender + firstSegIdx;
                    uint64_t firstSlotOffset = slotAreaBase + static_cast<uint64_t>(firstSlotIdx) * tokenSlotSize_;

                    LocalCopy(config_.dstGM + static_cast<uint64_t>(pipelineBase + pipelineReceived) * config_.rowBytes,
                              localWinBase_ + firstSlotOffset,
                              static_cast<uint64_t>(totalTokens) * config_.rowBytes); // 读槽数据到recvGrad
                    slotsRead = batchSlots;
                    tokensRead = totalTokens;
                } else {
                    // 否则while分开处理
                    while (slotsRead < batchSlots) {
                        uint32_t segIdx = (localReadCnt + slotsRead) % slotsPerSender;
                        uint32_t slotGlobalIdx = pipelineIdx * slotsPerSender + segIdx;
                        uint64_t slotOffset = slotAreaBase + static_cast<uint64_t>(slotGlobalIdx) * tokenSlotSize_;

                        uint32_t thisTokens = maxTokensPerSlot_;
                        if (tokensRead + thisTokens > remaining) {
                            thisTokens = remaining - tokensRead;
                        }

                        LocalCopy(config_.dstGM + static_cast<uint64_t>(pipelineBase + pipelineReceived + tokensRead) *
                                                      config_.rowBytes,
                                  localWinBase_ + slotOffset, static_cast<uint64_t>(thisTokens) * config_.rowBytes);

                        tokensRead += thisTokens;
                        slotsRead++; // 一个一个slot挨着处理
                    }
                }

                localReadCnt += slotsRead; // recv消费值更新
                pipelineReceived += tokensRead;
                WriteLocalCounter(
                    tokenReadOffset_, srcRank, pipelineIdx,
                    static_cast<int32_t>(localReadCnt)); // 记录recv消费值到本rank窗口的credit位置，便于发送卡读
            }
        }
    }

    __aicore__ inline void Drain()
    {
        for (uint32_t i = 0; i < pendingHandleCount_; i++) {
            int32_t ret = comm_->Drain(pendingHandles_[i]);
            ascendc_assert(ret == 0, "Drain failed, ret=%d, rankId=%u, jobIdx=%u", ret, config_.rankId, jobIndex_);
        }
        pendingHandleCount_ = 0;
    }

    __aicore__ inline void LocalCopy(GM_ADDR dst, GM_ADDR src, uint64_t len)
    {
        // 本地数据段拷贝: GM→UB→GM pingpong
        if (len == 0) {
            return;
        }

        GlobalTensor<uint8_t> srcGm;
        GlobalTensor<uint8_t> dstGm;
        srcGm.SetGlobalBuffer((__gm__ uint8_t*)src);
        dstGm.SetGlobalBuffer((__gm__ uint8_t*)dst);

        LocalTensor<uint8_t> buf0 = copyPing_;
        LocalTensor<uint8_t> buf1 = copyPong_;

        DataCopyPadExtParams<uint8_t> pad{false, 0, 0, 0};

        uint64_t off = 0;
        uint32_t tileIdx = 0;
        uint64_t pendingOff = 0;
        uint32_t pendingLen = 0;
        uint32_t pendingBufIdx = 0;
        bool hasPendingMte3 = false;

        // 延迟一拍流水: 本轮 MTE2 读入新块, 同时 MTE3 写出上一块(pending), 读写重叠
        while (off < len) {
            uint32_t thisLen =
                (len - off > ALLTOALLV_MAX_COPY_BYTES) ? ALLTOALLV_MAX_COPY_BYTES : static_cast<uint32_t>(len - off);
            uint32_t bufIdx = tileIdx % 2U;
            LocalTensor<uint8_t>& buf = (bufIdx == 0U) ? buf0 : buf1;

            if (tileIdx >= 2U) {
                AscendC::WaitFlag<HardEvent::MTE3_MTE2>(evtMte3ToMte2_[bufIdx]);
            }

            DataCopyExtParams mte2Params{1U, thisLen, 0U, 0U, 0U};
            DataCopyPad(buf, srcGm[off], mte2Params, pad);
            AscendC::SetFlag<HardEvent::MTE2_MTE3>(evtMte2ToMte3_[bufIdx]);

            if (hasPendingMte3) {
                AscendC::WaitFlag<HardEvent::MTE2_MTE3>(evtMte2ToMte3_[pendingBufIdx]);
                LocalTensor<uint8_t>& pendBuf = (pendingBufIdx == 0U) ? buf0 : buf1;
                DataCopyExtParams mte3Params{1U, pendingLen, 0U, 0U, 0U};
                DataCopyPad(dstGm[pendingOff], pendBuf, mte3Params);
                AscendC::SetFlag<HardEvent::MTE3_MTE2>(evtMte3ToMte2_[pendingBufIdx]);
            }

            pendingOff = off;
            pendingLen = thisLen;
            pendingBufIdx = bufIdx;
            hasPendingMte3 = true;

            off += thisLen;
            tileIdx++;
        }

        if (hasPendingMte3) {
            AscendC::WaitFlag<HardEvent::MTE2_MTE3>(evtMte2ToMte3_[pendingBufIdx]);
            LocalTensor<uint8_t>& pendBuf = (pendingBufIdx == 0U) ? buf0 : buf1;
            DataCopyExtParams mte3Params{1U, pendingLen, 0U, 0U, 0U};
            DataCopyPad(dstGm[pendingOff], pendBuf, mte3Params);
            AscendC::SetFlag<HardEvent::MTE3_MTE2>(evtMte3ToMte2_[pendingBufIdx]);
        }

        uint32_t last = (tileIdx - 1U) % 2U;
        AscendC::WaitFlag<HardEvent::MTE3_MTE2>(evtMte3ToMte2_[last]);
        if (tileIdx >= 2U) {
            AscendC::WaitFlag<HardEvent::MTE3_MTE2>(evtMte3ToMte2_[last ^ 1U]);
        }
    }

private:
    // 窗口(commBuffer): barrier 标志区 -> write 计数器区 -> read credit 区 -> 数据槽区
    __aicore__ inline void DeriveLayout()
    {
        uint64_t totalCounterEntries =
            static_cast<uint64_t>(config_.rankSize) * static_cast<uint64_t>(pipelinesPerRank_);
        uint64_t totalRanksOffset =
            totalCounterEntries * ALLTOALLV_COUNTER_SLOT_BYTES; // 每 rank 32B  rankSize×pipelines×32B
        // 设置了3个区域等宽，barrier 标志区，write 计数器区，read credit 区
        // 头部保留区(headReserveBytes, 如调用方的跨kernel屏障槽)不计入本模块布局
        tokenWriteOffset_ = config_.headReserveBytes + totalRanksOffset;
        tokenReadOffset_ = tokenWriteOffset_ + totalRanksOffset;
        tokenDataOffset_ = config_.headReserveBytes + ALLTOALLV_EQUAL_REGION_COUNT * totalRanksOffset;

        uint64_t tokenAreaRaw = (config_.winSize > tokenDataOffset_) ? (config_.winSize - tokenDataOffset_) : 0U;
        uint64_t tokenArea = tokenAreaRaw / config_.rowBytes * config_.rowBytes;
        tokenSlotSize_ = tokenArea / config_.rankSize / config_.numSlots;
        maxTokensPerSlot_ = static_cast<uint32_t>(tokenSlotSize_ / config_.rowBytes); // 一个slot最多装多少token
        if (maxTokensPerSlot_ == 0U) {
            maxTokensPerSlot_ = 1U;
        }
    }

    __aicore__ inline GM_ADDR GetRemoteWinAddr(uint32_t dstRank, uint64_t offset)
    {
        return (GM_ADDR)winAddrs_[dstRank] + offset;
    }

    __aicore__ inline uint64_t GetCommHandle(uint32_t dstRank, uint32_t pipelineIdx)
    {
        uint32_t channelIdx = pipelineIdx;
        if (channelIdx >= config_.channelsPerRank) {
            channelIdx = 0U;
        }
        return handles_[dstRank * config_.channelsPerRank + channelIdx];
    }

    // 读本卡窗口的 write 计数区（对端写过来的槽数）
    __aicore__ inline int32_t ReadLocalCounter(uint64_t counterOffset, uint32_t srcRank, uint32_t pipelineIdx)
    {
        __gm__ int32_t* counterAddr = reinterpret_cast<__gm__ int32_t*>(
            localWinBase_ + counterOffset +
            (static_cast<uint64_t>(srcRank) * pipelinesPerRank_ + pipelineIdx) * ALLTOALLV_COUNTER_SLOT_BYTES);
        return AscendC::ReadGmBypassDCache(counterAddr);
    }

    // 写本卡窗口的 read credit 区（本卡已消费的槽数）
    __aicore__ inline void WriteLocalCounter(uint64_t counterOffset, uint32_t peerRank, uint32_t pipelineIdx,
                                             int32_t value)
    {
        LocalTensor<int32_t> valLocal = statusUb_;
        valLocal.SetValue(0, value);
        AllToAllvSyncFunc<HardEvent::S_MTE3>(*pipe_);

        GM_ADDR counterAddr =
            localWinBase_ + counterOffset +
            (static_cast<uint64_t>(peerRank) * pipelinesPerRank_ + pipelineIdx) * ALLTOALLV_COUNTER_SLOT_BYTES;
        GlobalTensor<int32_t> counterGM;
        counterGM.SetGlobalBuffer((__gm__ int32_t*)counterAddr);
        DataCopyParams valParams = {1U, static_cast<uint16_t>(ALLTOALLV_UB_ALIGN), 0U, 0U};
        DataCopyPad(counterGM, valLocal, valParams);
        AllToAllvSyncFunc<HardEvent::MTE3_S>(*pipe_);
    }

    __aicore__ inline void PrefetchCreditCounter(uint32_t dstRank, uint32_t pipelineIdx)
    {
        if (creditReadInFlight_) {
            return;
        }

        GM_ADDR scratchAddr = config_.creditScratchGM + static_cast<uint64_t>(jobIndex_) * ALLTOALLV_UB_ALIGN;
        LocalTensor<int32_t> sentinelLocal = statusUb_;
        sentinelLocal.SetValue(0, ALLTOALLV_CREDIT_SENTINEL);
        AllToAllvSyncFunc<HardEvent::S_MTE3>(*pipe_);
        GlobalTensor<int32_t> scratchGM;
        scratchGM.SetGlobalBuffer((__gm__ int32_t*)scratchAddr);
        DataCopyParams sentinelParams = {1U, static_cast<uint16_t>(ALLTOALLV_UB_ALIGN), 0U, 0U};
        DataCopyPad(scratchGM, sentinelLocal, sentinelParams);
        AllToAllvSyncFunc<HardEvent::MTE3_S>(*pipe_);

        GM_ADDR remoteCounterAddr =
            GetRemoteWinAddr(dstRank, tokenReadOffset_) +
            (static_cast<uint64_t>(config_.rankId) * pipelinesPerRank_ + pipelineIdx) * ALLTOALLV_COUNTER_SLOT_BYTES;
        uint64_t handle = GetCommHandle(dstRank, pipelineIdx);
        int32_t ret = comm_->ReadNbi(handle, scratchAddr, remoteCounterAddr, sizeof(int32_t));
        ascendc_assert(ret == 0, "CreditRead launch failed, ret=%d, rankId=%u, dstRank=%u", ret, config_.rankId,
                       dstRank);
        creditReadInFlight_ = true;
    }

    __aicore__ inline int32_t CompleteCreditCounter()
    {
        GM_ADDR scratchAddr = config_.creditScratchGM + static_cast<uint64_t>(jobIndex_) * ALLTOALLV_UB_ALIGN;
        GlobalTensor<int32_t> scratchGM;
        scratchGM.SetGlobalBuffer((__gm__ int32_t*)scratchAddr);
        LocalTensor<int32_t> creditLocal = statusUb_;
        int32_t value = ALLTOALLV_CREDIT_SENTINEL;
        while (value == ALLTOALLV_CREDIT_SENTINEL) {
            DataCopyExtParams cpParams{1U, static_cast<uint32_t>(ALLTOALLV_UB_ALIGN), 0U, 0U, 0U};
            DataCopyPadExtParams<int32_t> cpPad{false, 0, 0, 0};
            DataCopyPad(creditLocal, scratchGM, cpParams, cpPad);
            AllToAllvSyncFunc<HardEvent::MTE2_S>(*pipe_);
            value = creditLocal.GetValue(0);
        }
        creditReadInFlight_ = false;
        return value;
    }

    __aicore__ inline void RetireCreditCounter()
    {
        if (!creditReadInFlight_) {
            return;
        }
        (void)CompleteCreditCounter();
    }

    __aicore__ inline void SendUnitRemote(uint32_t dstRank, uint32_t pipelineIdx, int32_t sendCount, int64_t sdispl)
    {
        uint64_t handle = GetCommHandle(dstRank, pipelineIdx);
        uint32_t totalSent = 0;
        uint32_t localWriteCnt = 0; // 本端已经发送的槽数
        int32_t remoteReadCnt = 0;  // 对端消费的槽数
        uint32_t slotsPerSender = config_.numSlots / pipelinesPerRank_;
        // 当已经发送的小于需要发送的任务，就while循环
        while (totalSent < static_cast<uint32_t>(sendCount)) {
            if (!creditReadInFlight_ && localWriteCnt + 1U >= static_cast<uint32_t>(remoteReadCnt) + slotsPerSender) {
                // 即将满，先读credit，异步发起一次“读对端窗口 credit 计数格”的 RDMA（ReadNbi），结果落到 scratch
                // 后立刻返回不等完成
                PrefetchCreditCounter(dstRank, pipelineIdx);
            }
            while (localWriteCnt >= static_cast<uint32_t>(remoteReadCnt) &&
                   localWriteCnt - static_cast<uint32_t>(remoteReadCnt) >= slotsPerSender) {
                PrefetchCreditCounter(dstRank, pipelineIdx);
                remoteReadCnt = CompleteCreditCounter(); // 等待credit读完成
            }

            uint32_t remaining = static_cast<uint32_t>(sendCount) - totalSent;
            uint32_t chunkLen = (remaining > maxTokensPerSlot_) ? maxTokensPerSlot_ : remaining;
            ascendc_assert(chunkLen != 0U, "SendUnitRemote chunkLen is 0");

            uint64_t slotBase = static_cast<uint64_t>(config_.rankId) * config_.numSlots * tokenSlotSize_;
            uint64_t slotIdx = static_cast<uint64_t>(pipelineIdx) * slotsPerSender +
                               (static_cast<uint64_t>(localWriteCnt) % slotsPerSender);
            uint64_t slotOffset = tokenDataOffset_ + slotBase + slotIdx * tokenSlotSize_;
            GM_ADDR remoteSlotAddr = GetRemoteWinAddr(dstRank, slotOffset);
            GM_ADDR srcAddr = config_.srcGM + static_cast<uint64_t>(sdispl + totalSent) * config_.rowBytes;
            uint64_t dataBytes = static_cast<uint64_t>(chunkLen) * config_.rowBytes;

            GM_ADDR remoteCounterAddr = GetRemoteWinAddr(dstRank, tokenWriteOffset_) +
                                        (static_cast<uint64_t>(config_.rankId) * pipelinesPerRank_ + pipelineIdx) *
                                            ALLTOALLV_COUNTER_SLOT_BYTES;
            int32_t ret = comm_->WriteWithNotifyNbi(handle, remoteSlotAddr, srcAddr, dataBytes, remoteCounterAddr,
                                                    static_cast<uint64_t>(localWriteCnt + 1));
            if (ret != 0) {
                for (uint32_t i = 0; i < ALLTOALLV_RETRY_COUNT; i++) {
                    ret = comm_->WriteWithNotifyNbi(handle, remoteSlotAddr, srcAddr, dataBytes, remoteCounterAddr,
                                                    static_cast<uint64_t>(localWriteCnt + 1));
                    if (ret == 0) {
                        break;
                    }
                }
                ascendc_assert(ret == 0, "WriteWithNotifyNbi failed after %u retries, ret=%d, rankId=%u, dstRank=%u",
                               ALLTOALLV_RETRY_COUNT, ret, config_.rankId, dstRank);
            }

            localWriteCnt++;
            totalSent += chunkLen;
        }
        RetireCreditCounter(); // 最后一次credit可能预取成功导致没有执行complete，此处complete兜底
        // 句柄暂存数组满时排空
        if (pendingHandleCount_ >= ALLTOALLV_MAX_PENDING) {
            for (uint32_t i = 0; i < pendingHandleCount_; i++) {
                int32_t ret = comm_->Drain(pendingHandles_[i]);
                ascendc_assert(ret == 0, "Drain failed, ret=%d, rankId=%u, jobIdx=%u", ret, config_.rankId, jobIndex_);
            }
            pendingHandleCount_ = 0;
        }
        ascendc_assert(pendingHandleCount_ < ALLTOALLV_MAX_PENDING, "pendingHandles overflow: count=%u, max=%u",
                       pendingHandleCount_, ALLTOALLV_MAX_PENDING);
        pendingHandles_[pendingHandleCount_] = handle;
        pendingHandleCount_++;
    }

    AllToAllvConfig config_;
    __gm__ const uint64_t* handles_{nullptr};
    __gm__ const uint64_t* winAddrs_{nullptr};
    AscendC::Hcomm<COMM_PROTOCOL_UBC_CTP>* comm_{nullptr};
    AscendC::LocalTensor<int32_t> statusUb_;
    AscendC::LocalTensor<uint8_t> copyPing_;
    AscendC::LocalTensor<uint8_t> copyPong_;
    AscendC::TPipe* pipe_{nullptr};
    GM_ADDR localWinBase_{0};

    uint32_t jobIndex_{0};
    uint32_t pipelinesPerRank_{1};

    uint64_t tokenWriteOffset_{0};
    uint64_t tokenReadOffset_{0};
    uint64_t tokenDataOffset_{0};
    uint64_t tokenSlotSize_{0};
    uint32_t maxTokensPerSlot_{1};

    uint64_t pendingHandles_[ALLTOALLV_MAX_PENDING]{0};
    uint32_t pendingHandleCount_{0};
    bool creditReadInFlight_{false};
    int32_t evtMte2ToMte3_[2] = {0, 0};
    int32_t evtMte3ToMte2_[2] = {0, 0};

public:
    uint32_t sendJobCnt_{1};
    uint32_t recvJobCnt_{1};
};

} // namespace Mc2Kernel

#endif
