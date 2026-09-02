/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mutex_buffer_manager.h
 * \brief MutexBuffer内存管理
 */
#ifndef MUTEX_BUFFER_MANAGER_H
#define MUTEX_BUFFER_MANAGER_H

#include "mutex_buffer.h"

namespace fa_base_matmul {
template <BufferType bufferType>
class MutexBufferManager {
    using TensorType = LocalTensor<uint8_t>;

public:
    __aicore__ inline void Init(TPipe *pipe, uint32_t size)
    {
        TBuf<BufferInfo<bufferType>::Position> tbuf;
        pipe->InitBuffer(tbuf, size);
        mem_ = tbuf.template Get<uint8_t>();
    }

    template <SyncType syncType = SyncType::INNER_CORE_SYNC>
    __aicore__ inline MutexBuffer<bufferType, syncType> AllocBuffer(uint32_t size)
    {
        TensorType temp = mem_[offset_];
        offset_ += size;
        return MutexBuffer<bufferType, syncType>(temp, size);
    }

    template <SyncType syncType = SyncType::INNER_CORE_SYNC>
    __aicore__ inline void FreeBuffer(MutexBuffer<bufferType, syncType> &buffer)
    {}

private:
    uint32_t offset_ = 0;
    TensorType mem_;
};
} // namespace fa_base_matmul
#endif
