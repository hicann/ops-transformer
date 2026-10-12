/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include "kernel_operator.h"

namespace Apace {
// Materialize [rank, row, bytesPerRank] as [row, rank, bytesPerRank].
// FP4 uses packed bytes unchanged; the caller converts logical K to bytes once.
// No HCCL or cross-core operations: all workers must be notified before CopyTile.
class AllToAllPermute {
public:
    static constexpr uint32_t BUFFER_BYTES = 248 * 1024;

    __aicore__ inline void Init(GM_ADDR rawInput, GM_ADDR output, uint64_t rows, uint32_t ranks, uint64_t bytesPerRank,
                                uint32_t workerId, uint32_t workerCount, AscendC::TPipe* pipe)
    {
        // The caller provides valid buffers, a 32-byte-aligned rank segment and a MIX_AIC_1_2 worker assignment.
        rows_ = rows;
        ranks_ = ranks;
        bytesPerRank_ = bytesPerRank;
        workerId_ = workerId;
        workerCount_ = workerCount;
        chunkCount_ = (bytesPerRank - 1) / BUFFER_BYTES + 1;
        input_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(rawInput));
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(output));
        pipe_ = pipe;
        pipe_->InitBuffer(buffer_, BUFFER_BYTES);
        inputReady_ = pipe_->AllocEventID<AscendC::HardEvent::MTE2_MTE3>();
        bufferFree_ = pipe_->AllocEventID<AscendC::HardEvent::MTE3_MTE2>();
        tileDone_ = pipe_->AllocEventID<AscendC::HardEvent::MTE3_S>();
    }

    __aicore__ inline void CopyTile(uint64_t rowStart, uint64_t rowCount)
    {
        const uint64_t taskCount = rowCount * ranks_ * chunkCount_;
        auto local = buffer_.Get<uint8_t>();
        for (uint64_t task = workerId_; task < taskCount; task += workerCount_) {
            const uint64_t chunk = task % chunkCount_;
            const uint64_t segment = task / chunkCount_;
            const uint64_t rank = segment % ranks_;
            const uint64_t row = rowStart + segment / ranks_;
            const uint64_t byteStart = chunk * BUFFER_BYTES;
            const uint64_t remaining = bytesPerRank_ - byteStart;
            const uint32_t count = static_cast<uint32_t>(remaining < BUFFER_BYTES ? remaining : BUFFER_BYTES);
            // Source rank stride is the complete M, including when this is a short tail tile.
            const uint64_t src = (rank * rows_ + row) * bytesPerRank_ + byteStart;
            const uint64_t dst = (row * ranks_ + rank) * bytesPerRank_ + byteStart;
            AscendC::DataCopy(local, input_[src], count);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(inputReady_);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(inputReady_);
            AscendC::DataCopy(output_[dst], local, count);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(bufferFree_);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(bufferFree_);
        }
        // A DMA-to-DMA wait alone does not block the scalar loop. Drain MTE3 before
        // returning to the notification/ACK protocol and before releasing events.
        if (workerId_ < taskCount) {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(tileDone_);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(tileDone_);
        }
    }

    __aicore__ inline void Finish()
    {
        pipe_->ReleaseEventID<AscendC::HardEvent::MTE2_MTE3>(inputReady_);
        pipe_->ReleaseEventID<AscendC::HardEvent::MTE3_MTE2>(bufferFree_);
        pipe_->ReleaseEventID<AscendC::HardEvent::MTE3_S>(tileDone_);
    }

private:
    AscendC::GlobalTensor<uint8_t> input_;
    AscendC::GlobalTensor<uint8_t> output_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> buffer_;
    AscendC::TPipe* pipe_{nullptr};
    AscendC::TEventID inputReady_{};
    AscendC::TEventID bufferFree_{};
    AscendC::TEventID tileDone_{};
    uint64_t rows_{0};
    uint64_t bytesPerRank_{0};
    uint64_t chunkCount_{0};
    uint32_t ranks_{0};
    uint32_t workerId_{0};
    uint32_t workerCount_{0};
};
} // namespace Apace
