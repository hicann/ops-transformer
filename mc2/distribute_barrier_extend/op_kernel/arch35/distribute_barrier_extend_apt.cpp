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
 * \file distribute_barrier_extend_apt.cpp
 * \brief APT (arch35) kernel entry for DistributeBarrierExtend.
 */

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#define BARRIER_SYSTEM_CLOCK_MIN_VERSION 90200000
#if ASC_DEVKIT_VERSION_NUM >= BARRIER_SYSTEM_CLOCK_MIN_VERSION && \
    (!defined(ASCENDC_CPU_DEBUG) || ASCENDC_CPU_DEBUG != 1)
#include "c_api/utils/sys_var.h"
#endif
#include "../../../distribute_barrier/op_kernel/distribute_barrier_tiling.h"
#include "../../../distribute_barrier/op_kernel/distribute_barrier.h"
#if __NPU_ARCH__ == 3510 && ASC_DEVKIT_VERSION_NUM >= MC2_BARRIER_HCOMM_MIN_VERSION
#include "adv_api/hcomm/hcomm.h"
#endif

namespace NsDistributeBarrierExtend {
using namespace AscendC;
using namespace Mc2Barrier;
constexpr uint32_t ELASTIC_CHUNK_ELEMS = 1024U;
constexpr uint32_t COMPLETION_OFFSET = SLOT_BYTES / 2U;
constexpr uint32_t MASK_WORD_BITS = sizeof(uint32_t) * 8U;
constexpr uint32_t BARRIER_PHASE_COUNT = 2U;

using DistributeBarrierImpl::ELASTIC_METAINFO_OFFSET;

__aicore__ inline uint64_t GetTimeUs()
{
#if ASC_DEVKIT_VERSION_NUM >= BARRIER_SYSTEM_CLOCK_MIN_VERSION && \
    (!defined(ASCENDC_CPU_DEBUG) || ASCENDC_CPU_DEBUG != 1)
    return static_cast<uint64_t>(asc_get_system_clock());
#else
    return static_cast<uint64_t>(GetSystemCycle()) / 1000UL;
#endif
}

__aicore__ inline void CheckBarrierCondition(bool condition)
{
    if (!condition) {
        AscendC::Trap();
    }
}

class DistributeBarrier {
public:
    __aicore__ inline void Init(GM_ADDR context, GM_ADDR timeout, GM_ADDR elasticInfo, TPipe* pipe,
                                const DistributeBarrierTilingData* tiling)
    {
        context_ = reinterpret_cast<__gm__ Context*>(context);
        CheckContext(tiling);
        localSize_ = context_->localSize;
        railSize_ = context_->railSize;
        source_ = reinterpret_cast<GM_ADDR>(context_->window) + SourceOffset(localSize_, railSize_);
        aivId_ = GetBlockIdx();
        workerNum_ = GetBlockNum() - 1U;
        pipe->InitBuffer(flagBuf_, BLOCK_BYTES);
        flag_ = flagBuf_.Get<float>();
        Duplicate(flag_, 1.0F, BLOCK_BYTES / sizeof(float));
        SyncFunc<HardEvent::V_MTE3>();
        pipe->InitBuffer(epochBuf_, BLOCK_BYTES);
        epochFlag_ = epochBuf_.Get<uint64_t>();
        InitComm(pipe);
        pipe->InitBuffer(waitBuf_, WAIT_BATCH_SLOTS * BLOCK_BYTES);
        wait_ = waitBuf_.Get<float>();
        hasTimeout_ = tiling->distributeBarrierInfo.isInputTimeOut;
        if (hasTimeout_) {
            InitTimeout(timeout);
        }
        begin_ = GetTimeUs();
        if (tiling->distributeBarrierInfo.isInputElasticInfo) {
            InitElasticInfo(elasticInfo, pipe);
        }
        InitState();
        if (!isElastic_) {
            begin_ = GetTimeUs();
        }
    }

    __aicore__ inline void Process()
    {
        Barrier(0U);
        if (railSize_ > 1) {
            Barrier(1U);
        }
        if (isElastic_) {
            InitState();
            Barrier(0U);
            if (railSize_ > 1) {
                Barrier(1U);
            }
        }
    }

private:
    __aicore__ inline void CheckContext(const DistributeBarrierTilingData* tiling)
    {
        const auto& info = tiling->distributeBarrierInfo;
        CheckBarrierCondition(context_->magic == MAGIC && context_->version == VERSION);
        const uint32_t local = context_->localSize;
        const uint32_t rail = context_->railSize;
        CheckBarrierCondition(context_->worldSize == info.worldSize &&
                              static_cast<uint64_t>(local) * rail == info.worldSize);
        CheckBarrierCondition(ContextElems(local, rail) == info.totalWinSize);
    }

    __aicore__ inline void InitComm(TPipe* pipe)
    {
#if __NPU_ARCH__ == 3510 && ASC_DEVKIT_VERSION_NUM >= MC2_BARRIER_HCOMM_MIN_VERSION
        if ((context_->protocolMask & (1U << static_cast<uint32_t>(Protocol::URMA))) != 0) {
            pipe->InitBuffer(urmaBuf_, HCOMM_BYTES);
            CheckBarrierCondition(urma_.Init(urmaBuf_.Get<uint8_t>(), HCOMM_BYTES) == 0);
        }
        if ((context_->protocolMask & (1U << static_cast<uint32_t>(Protocol::ROCE))) != 0) {
            pipe->InitBuffer(roceBuf_, HCOMM_BYTES);
            CheckBarrierCondition(roce_.Init(roceBuf_.Get<uint8_t>(), HCOMM_BYTES) == 0);
        }
#else
        (void)pipe;
        CheckBarrierCondition((context_->protocolMask & ~1U) == 0U);
#endif
    }

    __aicore__ inline void InitTimeout(GM_ADDR timeout)
    {
        GlobalTensor<int32_t> value;
        value.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(timeout));
        const int32_t duration = value.GetValue(0);
        CheckBarrierCondition(duration > 0);
        timeout_ = static_cast<uint32_t>(duration);
    }

    __aicore__ inline void InitElasticInfo(GM_ADDR elasticInfo, TPipe* pipe)
    {
        elasticGM_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(elasticInfo));
        DataCacheCleanAndInvalid<int32_t, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(elasticGM_);
        auto enabled = wait_.ReinterpretCast<int32_t>();
        DataCopyExtParams copy{1U, ELASTIC_METAINFO_OFFSET * sizeof(int32_t), 0U, 0U, 0U};
        DataCopyPadExtParams<int32_t> pad{false, 0U, 0U, 0U};
        DataCopyPad(enabled, elasticGM_, copy, pad);
        SyncFunc<HardEvent::MTE2_S>();
        const int32_t flag = enabled.GetValue(0);
        const int32_t active = enabled.GetValue(1);
        SyncFunc<HardEvent::S_MTE2>();
        CheckBarrierCondition(flag == 0 || flag == 1);
        isElastic_ = flag == 1;
        if (!isElastic_) {
            return;
        }
        CheckBarrierCondition(context_->worldSize <= MAX_RANKS && active > 0 &&
                              active <= static_cast<int32_t>(context_->worldSize));
        CheckBarrierCondition(railSize_ == 1 || context_->worldSize - static_cast<uint32_t>(active) < localSize_);
        if (aivId_ == 0) {
            CheckElasticRanks(active);
        } else {
            InitPeerMask(pipe);
        }
        SyncAll<true>();
    }

    __aicore__ inline void CheckElasticRanks(int32_t active)
    {
        const uint32_t world = context_->worldSize;
        const int32_t self = elasticGM_.GetValue(ELASTIC_METAINFO_OFFSET + context_->rank);
        CheckBarrierCondition(self >= 0 && self < active);
        auto chunk = wait_.ReinterpretCast<int32_t>();
        int32_t count = 0;
        for (uint32_t first = 0; first < world; first += ELASTIC_CHUNK_ELEMS) {
            const uint32_t size = world - first < ELASTIC_CHUNK_ELEMS ? world - first : ELASTIC_CHUNK_ELEMS;
            DataCopyExtParams copy{1U, static_cast<uint32_t>(size * sizeof(int32_t)), 0U, 0U, 0U};
            DataCopyPadExtParams<int32_t> pad{false, 0U, 0U, 0U};
            DataCopyPad(chunk, elasticGM_[ELASTIC_METAINFO_OFFSET + first], copy, pad);
            SyncFunc<HardEvent::MTE2_S>();
            for (uint32_t i = 0; i < size; ++i) {
                const int32_t logical = chunk.GetValue(i);
                if (logical == -1) {
                    continue;
                }
                CheckBarrierCondition(logical >= 0 && logical < active);
                CheckBarrierCondition(elasticGM_.GetValue(ELASTIC_METAINFO_OFFSET + world + logical) ==
                                      static_cast<int32_t>(first + i));
                ++count;
            }
            SyncFunc<HardEvent::S_MTE2>();
            CheckBarrierCondition(!hasTimeout_ || GetTimeUs() - begin_ < timeout_);
        }
        CheckBarrierCondition(count == active);
    }

    __aicore__ inline void InitPeerMask(TPipe* pipe)
    {
        const uint32_t bytes = ElasticMaskBytes(context_->worldSize, workerNum_);
        pipe->InitBuffer(elasticBuf_, bytes);
        activePeers_ = elasticBuf_.Get<uint32_t>();
        Duplicate(activePeers_, 0U, bytes / sizeof(uint32_t));
        SyncFunc<HardEvent::V_S>();
        const auto* peers = reinterpret_cast<__gm__ Peer*>(context_ + 1);
        uint32_t bit = 0;
        for (uint32_t phase = 0; phase < BARRIER_PHASE_COUNT; ++phase) {
            const uint32_t count = phase == 0 ? localSize_ : railSize_;
            const uint32_t first = WorkerFirst(aivId_ - 1U, count);
            const uint32_t end = WorkerFirst(aivId_, count);
            for (uint32_t i = first; i < end; ++i, ++bit) {
                const uint32_t rank = peers[(phase == 0 ? 0 : localSize_) + i].rank;
                CheckBarrierCondition(rank < context_->worldSize);
                if (elasticGM_.GetValue(ELASTIC_METAINFO_OFFSET + rank) >= 0) {
                    activePeers_.SetValue(bit / MASK_WORD_BITS,
                                          activePeers_.GetValue(bit / MASK_WORD_BITS) | (1U << (bit % MASK_WORD_BITS)));
                }
            }
            CheckBarrierCondition(!hasTimeout_ || GetTimeUs() - begin_ < timeout_);
        }
    }

    __aicore__ inline bool IsActive(uint32_t slot, uint32_t phase)
    {
        if (!isElastic_) {
            return true;
        }
        const uint32_t count = phase == 0 ? localSize_ : railSize_;
        uint32_t bit = slot - WorkerFirst(aivId_ - 1U, count);
        if (phase != 0) {
            bit += WorkerFirst(aivId_, localSize_) - WorkerFirst(aivId_ - 1U, localSize_);
        }
        return (activePeers_.GetValue(bit / MASK_WORD_BITS) & (1U << (bit % MASK_WORD_BITS))) != 0U;
    }

    __aicore__ inline void InitState()
    {
        GlobalTensor<uint64_t> sequence;
        sequence.SetGlobalBuffer(
            reinterpret_cast<__gm__ uint64_t*>(context_->window + SequenceOffset(localSize_, railSize_)));
        if (aivId_ == 0) {
            DataCacheCleanAndInvalid<uint64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(sequence);
            const uint64_t current = sequence.GetValue(0);
            const uint64_t next = isElastic_ ? current ^ 1U : current + 1U;
            CheckBarrierCondition(isElastic_ || next != 0U);
            sequence.SetValue(0, next);
            DataCacheCleanAndInvalid<uint64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(sequence);
            InitEpochFlag(isElastic_ ? 1U : next);
            GlobalTensor<uint32_t> source;
            source.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(source_));
            const auto signal = isElastic_ ? flag_.ReinterpretCast<uint32_t>() : epochFlag_.ReinterpretCast<uint32_t>();
            DataCopy(source, signal, BLOCK_BYTES / sizeof(uint32_t));
            SyncFunc<HardEvent::MTE3_S>();
        }
        SyncAll<true>();
        DataCacheCleanAndInvalid<uint64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(sequence);
        const uint64_t current = sequence.GetValue(0);
        state_ = current & 1U;
        if (aivId_ != 0) {
            InitEpochFlag(isElastic_ ? 1U : current);
        }
    }

    __aicore__ inline void InitEpochFlag(uint64_t epoch)
    {
        epoch_ = epoch;
        for (uint32_t i = 0; i < BLOCK_BYTES / sizeof(uint64_t); ++i) {
            epochFlag_.SetValue(i, epoch);
        }
        SyncFunc<HardEvent::S_MTE3>();
    }

    __aicore__ inline GM_ADDR FlagAddress(uint64_t window, uint32_t phase, uint32_t slot)
    {
        return reinterpret_cast<GM_ADDR>(window) + state_ * BankBytes(localSize_, railSize_) +
               PhaseOffset(phase, localSize_) + slot * SLOT_BYTES;
    }

    __aicore__ inline void Send(const __gm__ Peer& peer, uint32_t phase)
    {
        const uint32_t slot = phase == 0 ? context_->localSlot : context_->railSlot;
        GM_ADDR destination = FlagAddress(peer.window, phase, slot);
        if (peer.protocol == Protocol::UBMEM) {
            GlobalTensor<uint32_t> remote;
            remote.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(destination));
            const auto signal = isElastic_ ? flag_.ReinterpretCast<uint32_t>() : epochFlag_.ReinterpretCast<uint32_t>();
            DataCopy(remote, signal, BLOCK_BYTES / sizeof(uint32_t));
        } else {
            WriteRemote(peer, destination);
        }
    }

    __aicore__ inline void WriteRemote(const __gm__ Peer& peer, GM_ADDR destination)
    {
#if __NPU_ARCH__ == 3510 && ASC_DEVKIT_VERSION_NUM >= MC2_BARRIER_HCOMM_MIN_VERSION
        const bool isRoce = peer.protocol == Protocol::ROCE;
        if (!isRoce) {
            CheckBarrierCondition(urma_.Lock(peer.handle) == 0);
        }
        int32_t ret = isRoce ? roce_.WriteNbi<true, PIPE_S, PIPE_MTE3>(peer.handle, destination, source_, BLOCK_BYTES) :
                               urma_.WriteNbi<true, PIPE_S, PIPE_MTE3>(peer.handle, destination, source_, BLOCK_BYTES);
        if (ret == 0) {
            ret = isRoce ? roce_.Drain(peer.handle) : urma_.Drain(peer.handle);
        }
        const int32_t unlockRet = isRoce ? 0 : urma_.Unlock(peer.handle);
        CheckBarrierCondition(ret == 0 && unlockRet == 0);
#else
        (void)peer;
        (void)destination;
        CheckBarrierCondition(false);
#endif
    }

    __aicore__ inline bool FlagsReady(uint32_t first, uint32_t count, uint32_t phase)
    {
        if (!isElastic_) {
            const auto flags = wait_.ReinterpretCast<uint64_t>();
            for (uint32_t i = 0; i < count * BLOCK_BYTES / sizeof(uint64_t); ++i) {
                if (flags.GetValue(i) < epoch_) {
                    return false;
                }
            }
            return true;
        }
        for (uint32_t i = 0; i < count; ++i) {
            if (IsActive(first + i, phase) && wait_.GetValue(i * BLOCK_BYTES / sizeof(float)) != 1.0F) {
                return false;
            }
        }
        return true;
    }

    __aicore__ inline void WaitAndClear(uint32_t first, uint32_t end, uint32_t phase)
    {
        GlobalTensor<float> flags;
        flags.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(FlagAddress(context_->window, phase, first)));
        uint32_t offset = 0;
        while (first + offset < end) {
            const uint32_t remaining = end - first - offset;
            const uint32_t count = remaining < WAIT_BATCH_SLOTS ? remaining : WAIT_BATCH_SLOTS;
            DataCopyExtParams copy{static_cast<uint16_t>(count), BLOCK_BYTES, SLOT_BYTES - BLOCK_BYTES, 0U, 0U};
            DataCopyPadExtParams<float> pad{false, 0U, 0U, 0.0F};
            bool ready = false;
            while (!ready) {
                DataCopyPad(wait_, flags[static_cast<uint64_t>(offset) * SLOT_BYTES / sizeof(float)], copy, pad);
                SyncFunc<HardEvent::MTE2_S>();
                ready = FlagsReady(first + offset, count, phase);
                SyncFunc<HardEvent::S_MTE2>();
            }
            SyncFunc<HardEvent::S_V>();
            Duplicate(wait_, 0.0F, count * BLOCK_BYTES / sizeof(float));
            SyncFunc<HardEvent::V_MTE3>();
            DataCopyExtParams clear{static_cast<uint16_t>(count), BLOCK_BYTES, 0U, SLOT_BYTES - BLOCK_BYTES, 0U};
            DataCopyPad(flags[static_cast<uint64_t>(offset) * SLOT_BYTES / sizeof(float)], wait_, clear);
            SyncFunc<HardEvent::MTE3_S>();
            offset += count;
        }
    }

    __aicore__ inline uint32_t WorkerFirst(uint32_t worker, uint32_t count)
    {
        const uint32_t extra = count % workerNum_;
        return worker * (count / workerNum_) + (worker < extra ? worker : extra);
    }

    __aicore__ inline void WaitWorkers(uint32_t count, uint32_t phase)
    {
        const uint32_t activeWorkers = count < workerNum_ ? count : workerNum_;
        for (uint32_t worker = 0; worker < activeWorkers; ++worker) {
            GlobalTensor<uint64_t> completed;
            completed.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t*>(
                FlagAddress(context_->window, phase, WorkerFirst(worker, count)) + COMPLETION_OFFSET));
            while (true) {
                DataCacheCleanAndInvalid<uint64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(completed);
                if (hasTimeout_) {
                    CheckBarrierCondition(GetTimeUs() - begin_ < timeout_);
                }
                if (completed.GetValue(0) == epoch_) {
                    break;
                }
            }
            completed.SetValue(0, 0U);
            DataCacheCleanAndInvalid<uint64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(completed);
        }
    }

    __aicore__ inline void WorkPeers(uint32_t phase)
    {
        const auto* peers = reinterpret_cast<__gm__ Peer*>(context_ + 1) + (phase == 0 ? 0 : localSize_);
        const uint32_t count = phase == 0 ? localSize_ : railSize_;
        const uint32_t worker = aivId_ - 1U;
        const uint32_t base = count / workerNum_;
        const uint32_t extra = count % workerNum_;
        const uint32_t first = WorkerFirst(worker, count);
        const uint32_t end = first + base + (worker < extra ? 1U : 0U);
        for (uint32_t i = first; i < end; ++i) {
            if (IsActive(i, phase)) {
                Send(peers[i], phase);
            }
        }
        SyncFunc<HardEvent::MTE3_S>();
        WaitAndClear(first, end, phase);
        if (first < end) {
            GlobalTensor<uint64_t> completed;
            completed.SetGlobalBuffer(
                reinterpret_cast<__gm__ uint64_t*>(FlagAddress(context_->window, phase, first) + COMPLETION_OFFSET));
            completed.SetValue(0, epoch_);
            DataCacheCleanAndInvalid<uint64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(completed);
        }
    }

    __aicore__ inline void Barrier(uint32_t phase)
    {
        if (aivId_ == 0U) {
            WaitWorkers(phase == 0U ? localSize_ : railSize_, phase);
        } else {
            WorkPeers(phase);
        }
        SyncAll<true>();
    }

    __gm__ Context* context_{nullptr};
    GM_ADDR source_{nullptr};
#if __NPU_ARCH__ == 3510 && ASC_DEVKIT_VERSION_NUM >= MC2_BARRIER_HCOMM_MIN_VERSION
    Hcomm<COMM_PROTOCOL_UBC_CTP> urma_;
    Hcomm<COMM_PROTOCOL_ROCE> roce_;
#endif
    TBuf<> flagBuf_;
    TBuf<> epochBuf_;
    TBuf<> urmaBuf_;
    TBuf<> roceBuf_;
    TBuf<> waitBuf_;
    TBuf<> elasticBuf_;
    LocalTensor<float> flag_;
    LocalTensor<uint64_t> epochFlag_;
    LocalTensor<float> wait_;
    GlobalTensor<int32_t> elasticGM_;
    LocalTensor<uint32_t> activePeers_;
    uint32_t aivId_{0};
    uint32_t workerNum_{0};
    uint32_t localSize_{0};
    uint32_t railSize_{0};
    uint32_t state_{0};
    uint64_t begin_{0};
    uint64_t epoch_{0};
    uint32_t timeout_{0};
    bool hasTimeout_{false};
    bool isElastic_{false};
};
} // namespace NsDistributeBarrierExtend

using namespace AscendC;

extern "C" __global__ __aicore__ void distribute_barrier_extend(GM_ADDR mc2Context, GM_ADDR xRef, GM_ADDR timeOut,
                                                                GM_ADDR elasticInfo, GM_ADDR xRefOut,
                                                                GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(DistributeBarrierTilingData);
    GET_TILING_DATA_WITH_STRUCT(DistributeBarrierTilingData, tilingData, tilingGM);
    TPipe pipe;
    if (tilingData.distributeBarrierInfo.isAuto) {
        NsDistributeBarrierExtend::DistributeBarrier op;
        op.Init(mc2Context, timeOut, elasticInfo, &pipe, &tilingData);
        op.Process();
    } else {
        DistributeBarrierImpl::DistributeBarrier<DTYPE_X_REF> op;
        op.Init(mc2Context, timeOut, elasticInfo, workspaceGM, &pipe, &tilingData);
        op.Process();
    }
}
