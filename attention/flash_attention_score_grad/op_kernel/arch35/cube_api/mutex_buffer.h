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
 * \file mutex_buffer.h
 * \brief 基于Mutex的buffer同步管理（替代event同步）
 */
#ifndef MUTEX_BUFFER_H
#define MUTEX_BUFFER_H

#include "../../../../common/op_kernel/attn_buffer.h"

namespace fa_base_matmul {

template <BufferType Type>
struct MutexBufferInfo {
    __aicore__ const static constexpr pipe_t ProdPipe()
    {
        if constexpr (Type == BufferType::L1) {
            return PIPE_MTE2;
        } else if constexpr (Type == BufferType::L0A || Type == BufferType::L0B || Type == BufferType::C2) {
            return PIPE_MTE1;
        } else if constexpr (Type == BufferType::L0C) {
            return PIPE_M;
        }
    }

    __aicore__ const static constexpr pipe_t ConsPipe()
    {
        if constexpr (Type == BufferType::L1) {
            return PIPE_MTE1;
        } else if constexpr (Type == BufferType::L0A || Type == BufferType::L0B || Type == BufferType::C2) {
            return PIPE_M;
        } else if constexpr (Type == BufferType::L0C) {
            return PIPE_FIX;
        }
    }

    static constexpr pipe_t ProdPipeVal = ProdPipe();
    static constexpr pipe_t ConsPipeVal = ConsPipe();
    static constexpr TPosition Position = BufferInfo<Type>::Position;
};

template <BufferType bufferType, SyncType syncType = SyncType::INNER_CORE_SYNC>
class MutexBuffer {
    // TensorType用于定义缓冲区的基础数据类型（此处为uint8_t），用于互斥缓冲区的原始存储。
    using TensorType = LocalTensor<uint8_t>;

    // TargetTensorType是一个模板，允许根据需要组合不同的数据类型，便于不同数据类型的缓冲需求。
    template <typename T>
    using TargetTensorType = LocalTensor<T>;

public:
    __aicore__ inline MutexBuffer() {}
    __aicore__ inline MutexBuffer(TensorType tensor, uint32_t size)
        : tensor_(tensor),
          size_(size)
    {}

    __aicore__ inline void Init()
    {
        if constexpr (syncType == SyncType::INNER_CORE_SYNC) {
            if ASCEND_IS_AIC {
                mutexId_ = AllocMutexID();
            }
        }
    }

    __aicore__ inline void UnInit()
    {
        if constexpr (syncType == SyncType::INNER_CORE_SYNC) {
            if ASCEND_IS_AIC {
                ReleaseMutexID(mutexId_);
            }
        }
    }

    template <pipe_t pipe>
    __aicore__ inline void Lock()
    {
        if constexpr (syncType == SyncType::INNER_CORE_SYNC) {
            if ASCEND_IS_AIC {
                AscendC::Mutex::Lock<pipe>(mutexId_);
            }
        }
    }

    template <pipe_t pipe>
    __aicore__ inline void Unlock()
    {
        if constexpr (syncType == SyncType::INNER_CORE_SYNC) {
            if ASCEND_IS_AIC {
                AscendC::Mutex::Unlock<pipe>(mutexId_);
            }
        }
    }

    __aicore__ inline void LockProd()
    {
        Lock<MutexBufferInfo<bufferType>::ProdPipeVal>();
    }

    __aicore__ inline void UnlockProd()
    {
        Unlock<MutexBufferInfo<bufferType>::ProdPipeVal>();
    }

    __aicore__ inline void LockCons()
    {
        Lock<MutexBufferInfo<bufferType>::ConsPipeVal>();
    }

    __aicore__ inline void UnlockCons()
    {
        Unlock<MutexBufferInfo<bufferType>::ConsPipeVal>();
    }

    template <HardEvent EventType>
    __aicore__ inline void Wait() = delete;

    template <HardEvent EventType>
    __aicore__ inline void Set() = delete;

    template <typename T>
    __aicore__ inline TargetTensorType<T> GetTensor()
    {
        return tensor_.template ReinterpretCast<T>();
    }

    template <typename T>
    __aicore__ inline TargetTensorType<T> GetTensor(uint64_t startindex)
    {
        TargetTensorType<T> tmpTensor = tensor_.template ReinterpretCast<T>();
        return tmpTensor[startindex];
    }

private:
    TensorType tensor_;
    uint32_t size_ = 0;
    MutexID mutexId_ = 0;
};
} // namespace fa_base_matmul
#endif
