/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QUANT_FLASH_ATTN_SCHEDULER_INT8_H_
#define QUANT_FLASH_ATTN_SCHEDULER_INT8_H_
#include "kernel_operator.h"
#include "quant_flash_attn_tiling_data_int8.h"
#include "quant_flash_attn_workload_int8.h"

namespace BaseApi::Int8GqaTnd {
// Actual lengths remain on device. No per-token workspace or host dereference is needed to compact the packed task
// grid and balance its causal KV work.
class TndTaskScheduler {
    using Sequence = QfaSequenceDescriptor;

public:
    __aicore__ inline void Init(const QuantFlashAttnTilingDataInt8& t, GM_ADDR cq, GM_ADDR ck, uint32_t core,
                                GM_ADDR tiling)
    {
        t_ = t;
        cq_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(cq));
        ck_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(ck));
        const auto* payload = reinterpret_cast<const __gm__ QuantFlashAttnTilingPayloadInt8*>(tiling);
        // Cache up to 32 immutable descriptors; larger batches use the same GM reader.
        descriptorsCached_ = false;
        if (t_.batchSize <= DESCRIPTOR_CAPACITY) {
            for (uint32_t b = 0U; b < t_.batchSize; ++b)
                descriptors_[b] = ReadSequence(b);
            descriptorsCached_ = true;
        }
        seq_ = ReadSequence(0U);
        // Mirror pairs have constant work; use them only with an equal pair count per core.
        const uint64_t pairs = seq_.tasks / QFA_MIRRORED_TASK_PAIR_SIZE;
        if (t_.batchSize == 1U && t_.maskMode != 0U && seq_.sq == seq_.sk && seq_.sq > 0U &&
            seq_.sq % t_.blockN == 0U && static_cast<uint64_t>(seq_.sq) * t_.groupSize % t_.blockM == 0U &&
            static_cast<uint64_t>(t_.blockN) * t_.groupSize % t_.blockM == 0U &&
            seq_.tiles % QFA_MIRRORED_TASK_PAIR_SIZE == 0U && pairs % t_.usedCoreNum == 0U) {
            paired_ = true;
            begin_ = QFA_MIRRORED_TASK_PAIR_SIZE * (pairs * core / t_.usedCoreNum);
            end_ = QFA_MIRRORED_TASK_PAIR_SIZE * (pairs * (core + 1U) / t_.usedCoreNum);
            return;
        }
        if (t_.batchSize == 1U && payload->useCoreRanges != QFA_CORE_RANGES_NONE && seq_.sq == t_.seqLenQ &&
            seq_.sk == t_.seqLenKv) {
            mapped_ = payload->useCoreRanges == QFA_CORE_RANGES_TASK_MAP;
            if (mapped_)
                tasks_.SetGlobalBuffer(
                    reinterpret_cast<__gm__ QuantFlashAttnTilingQBlockPayloadInt8*>(tiling)->taskIds);
            begin_ = payload->coreTaskBegin[core];
            end_ = payload->coreTaskBegin[core + 1U];
            return;
        }
        QfaTaskRangePlanner<TndTaskScheduler> planner(
            {t_.batchSize, t_.usedCoreNum, t_.groupSize, t_.blockM, t_.blockN, t_.maskMode}, *this);
        planner.Init(core);
        begin_ = planner.Begin();
        end_ = planner.End();
        seq_ = ReadSequence(0U);
    }

    __aicore__ inline uint64_t Begin() const
    {
        return begin_;
    }
    __aicore__ inline uint64_t End() const
    {
        return end_;
    }
    __aicore__ inline bool Decode(uint64_t task, uint32_t& batch, uint32_t& head, uint32_t& mStart, uint32_t& sq,
                                  uint32_t& sk, uint32_t& qStart, uint32_t& kStart)
    {
        if (mapped_)
            task = tasks_.GetValue(task);
        while (task >= batchBase_ + seq_.tasks) {
            batchBase_ += seq_.tasks;
            if (++batch_ >= t_.batchSize)
                return false;
            seq_ = ReadSequence(batch_);
        }
        const uint32_t local = task - batchBase_;
        batch = batch_;
        if (paired_) {
            const uint32_t half = seq_.tiles / QFA_MIRRORED_TASK_PAIR_SIZE;
            head = (local / QFA_MIRRORED_TASK_PAIR_SIZE) / half;
            const uint32_t pair = (local / QFA_MIRRORED_TASK_PAIR_SIZE) % half;
            mStart = ((local & 1U) ? seq_.tiles - 1U - pair : pair) * t_.blockM;
        } else {
            head = local / seq_.tiles;
            mStart = (local % seq_.tiles) * t_.blockM;
        }
        sq = seq_.sq;
        sk = seq_.sk;
        qStart = seq_.qStart;
        kStart = seq_.kStart;
        return true;
    }

    __aicore__ inline Sequence ReadSequence(uint32_t b)
    {
        if (descriptorsCached_)
            return descriptors_[b];
        Sequence s;
        s.qStart = cq_.GetValue(b);
        s.kStart = ck_.GetValue(b);
        s.sq = cq_.GetValue(b + 1U) - s.qStart;
        s.sk = ck_.GetValue(b + 1U) - s.kStart;
        if (s.sq > t_.seqLenQ || s.sk > t_.seqLenKv || (s.sq > 0U && s.sk == 0U) ||
            static_cast<uint64_t>(s.qStart) + s.sq > t_.totalTokensQ ||
            static_cast<uint64_t>(s.kStart) + s.sk > t_.totalTokensKv)
            return Sequence{};
        s.tiles = (static_cast<uint64_t>(s.sq) * t_.groupSize + t_.blockM - 1U) / t_.blockM;
        s.tasks = s.tiles * t_.numHeadsKv;
        return s;
    }

private:
    static constexpr uint32_t DESCRIPTOR_CAPACITY = 32U;
    Sequence descriptors_[DESCRIPTOR_CAPACITY];
    bool descriptorsCached_ = false;
    // Scalar aggregate copies use 64-bit accesses; keep the private copy aligned without changing the tiling ABI.
    alignas(uint64_t) QuantFlashAttnTilingDataInt8 t_{};
    AscendC::GlobalTensor<int32_t> cq_, ck_;
    AscendC::GlobalTensor<uint32_t> tasks_;
    Sequence seq_{};
    uint64_t begin_ = 0U, end_ = 0U, batchBase_ = 0U;
    uint32_t batch_ = 0U;
    bool paired_ = false, mapped_ = false;
};

} // namespace BaseApi::Int8GqaTnd

#endif
