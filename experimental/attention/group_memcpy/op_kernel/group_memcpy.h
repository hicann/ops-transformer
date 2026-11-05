/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef GROUP_MEMCPY_KERNEL_H
#define GROUP_MEMCPY_KERNEL_H
#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"

namespace NsGroupMemcpy {
class KernelGroupMemcpy {
public:
    template <class TilingData>
    __aicore__ inline void Init(const TilingData& data)
    {
        pairCount_ = data.pairCount;
        bytesPerCopy_ = data.bytesPerCopy;
        tileBytes_ = data.tileBytes;
        if (bytesPerCopy_ != 0) {
            pipe_.InitBuffer(inQueue_, 1, tileBytes_);
            pipe_.InitBuffer(outQueue_, 1, tileBytes_);
        }
    }

    __aicore__ inline void Process(GM_ADDR src, GM_ADDR dst)
    {
        if (bytesPerCopy_ == 0) {
            return;
        }
        AscendC::ListTensorDesc srcList(reinterpret_cast<__gm__ void*>(src));
        AscendC::ListTensorDesc dstList(reinterpret_cast<__gm__ void*>(dst));
        // Test the task index BEFORE dereferencing either list (original tail-core bug).
        for (uint64_t task = AscendC::GetBlockIdx(); task < pairCount_; task += AscendC::GetBlockNum()) {
            inGm_.SetGlobalBuffer(srcList.GetDataPtr<__gm__ int8_t>(task), bytesPerCopy_);
            outGm_.SetGlobalBuffer(dstList.GetDataPtr<__gm__ int8_t>(task), bytesPerCopy_);
            for (uint64_t offset = 0; offset < bytesPerCopy_; offset += tileBytes_) {
                const uint64_t remaining = bytesPerCopy_ - offset;
                const uint32_t realBytes = static_cast<uint32_t>(remaining < tileBytes_ ? remaining : tileBytes_);
                const uint32_t alignedBytes = (realBytes + 63) / 64 * 64;
                CopyIn(offset, realBytes, alignedBytes);
                Compute(alignedBytes);
                CopyOut(offset, realBytes);
            }
        }
    }

private:
    __aicore__ inline void CopyIn(uint64_t offset, uint32_t bytes, uint32_t alignedBytes)
    {
        auto input = inQueue_.AllocTensor<int8_t>();
        AscendC::DataCopyExtParams params{1, bytes, 0, 0, 0};
        // Preserve the original 64-byte UB copy length, including its last
        // 32-byte block. Initialize it before DMA so Compute never reads an
        // uninitialized UB tail. Full tiles need no initialization.
        if (bytes != alignedBytes) {
            AscendC::Duplicate(input.ReinterpretCast<int16_t>(), static_cast<int16_t>(0), alignedBytes / 2);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
        const uint8_t rightPadding = static_cast<uint8_t>((32 - bytes % 32) % 32);
        AscendC::DataCopyPadExtParams<int8_t> padding{true, 0, rightPadding, 0};
        AscendC::DataCopyPad(input, inGm_[offset], params, padding);
        inQueue_.EnQue(input);
    }
    __aicore__ inline void Compute(uint32_t alignedBytes)
    {
        auto input = inQueue_.DeQue<int8_t>();
        auto output = outQueue_.AllocTensor<int8_t>();
        AscendC::DataCopy(output, input, alignedBytes);
        outQueue_.EnQue(output);
        inQueue_.FreeTensor(input);
    }
    __aicore__ inline void CopyOut(uint64_t offset, uint32_t bytes)
    {
        auto output = outQueue_.DeQue<int8_t>();
        AscendC::DataCopyExtParams params{1, bytes, 0, 0, 0};
        AscendC::DataCopyPad(outGm_[offset], output, params);
        outQueue_.FreeTensor(output);
    }
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQueue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outQueue_;
    AscendC::GlobalTensor<int8_t> inGm_;
    AscendC::GlobalTensor<int8_t> outGm_;
    uint64_t pairCount_ = 0;
    uint64_t bytesPerCopy_ = 0;
    uint64_t tileBytes_ = 0;
};
} // namespace NsGroupMemcpy
#endif
