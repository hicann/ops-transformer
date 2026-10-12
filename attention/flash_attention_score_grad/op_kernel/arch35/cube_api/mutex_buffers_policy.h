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
 * \file mutex_buffers_policy.h
 * \brief 基于Mutex的buffer多缓冲策略（替代event同步）
 */
#ifndef MUTEX_BUFFERS_POLICY_H
#define MUTEX_BUFFERS_POLICY_H

#include "mutex_buffer_manager.h"
#include "mutex_buffer.h"
#ifndef NUM_2
#define NUM_2 2
#endif
#ifndef NUM_3
#define NUM_3 3
#endif
#ifndef NUM_4
#define NUM_4 4
#endif

namespace fa_base_matmul {

namespace detail {
template <BufferType bufferType, SyncType syncType = SyncType::INNER_CORE_SYNC>
__aicore__ inline MutexBuffer<bufferType, syncType> AllocMutexBuf(MutexBufferManager<bufferType>& mgr, uint32_t size)
{
    return mgr.template AllocBuffer<syncType>(size);
}
} // namespace detail

template <BufferType bufferType, SyncType syncType = SyncType::INNER_CORE_SYNC>
class MutexBuffersPolicySingleBuffer {
public:
    __aicore__ inline void Init(MutexBufferManager<bufferType>& bufferManager, uint32_t size)
    {
        buffer_ = detail::AllocMutexBuf<bufferType, syncType>(bufferManager, size);
        buffer_.Init();
    }

    __aicore__ inline void Uninit(MutexBufferManager<bufferType>& bufferManager)
    {
        buffer_.UnInit();
    }

    __aicore__ inline MutexBuffer<bufferType, syncType>& Get()
    {
        return buffer_;
    }

    __aicore__ inline MutexBuffer<bufferType, syncType>& GetPre()
    {
        return Get();
    }

    __aicore__ inline MutexBuffer<bufferType, syncType>& GetReused()
    {
        return Get();
    }

private:
    MutexBuffer<bufferType, syncType> buffer_;
};

template <BufferType bufferType, SyncType syncType = SyncType::INNER_CORE_SYNC>
class MutexBuffersPolicyDB {
public:
    __aicore__ inline void Init(MutexBufferManager<bufferType>& bufferManager, uint32_t size)
    {
        ping_ = detail::AllocMutexBuf<bufferType, syncType>(bufferManager, size);
        pong_ = detail::AllocMutexBuf<bufferType, syncType>(bufferManager, size);

        ping_.Init();
        pong_.Init();
    }

    __aicore__ inline void Uninit(MutexBufferManager<bufferType>& bufferManager)
    {
        ping_.UnInit();
        pong_.UnInit();
    }

    __aicore__ inline MutexBuffer<bufferType, syncType>& Get()
    {
        if (flag1_) {
            flag1_ = 0;
            return ping_;
        } else {
            flag1_ = 1;
            return pong_;
        }
    }

    __aicore__ inline MutexBuffer<bufferType, syncType>& GetPre()
    {
        if (flag1_) {
            return pong_;
        } else {
            return ping_;
        }
    }

    __aicore__ inline MutexBuffer<bufferType, syncType>& GetReused()
    {
        if (flag2_ == 0) {
            flag2_ = 1;
            return pong_;
        } else {
            flag2_ = 0;
            return ping_;
        }
    }

    __aicore__ inline MutexBuffer<bufferType, syncType>& GetReused(bool isNextS2IdxNoChange)
    {
        if (isNextS2IdxNoChange) {
            if (flag2_ == 0) {
                return pong_;
            } else {
                return ping_;
            }
        } else {
            return GetReused();
        }
    }

private:
    MutexBuffer<bufferType, syncType> ping_;
    MutexBuffer<bufferType, syncType> pong_;
    uint32_t flag1_ = 0;
    uint32_t flag2_ = 0;
};

template <BufferType bufferType, SyncType syncType = SyncType::INNER_CORE_SYNC>
class MutexBuffersPolicy3buff {
public:
    __aicore__ inline void Init(MutexBufferManager<bufferType>& bufferManager, uint32_t size)
    {
        a_ = detail::AllocMutexBuf<bufferType, syncType>(bufferManager, size);
        b_ = detail::AllocMutexBuf<bufferType, syncType>(bufferManager, size);
        c_ = detail::AllocMutexBuf<bufferType, syncType>(bufferManager, size);

        a_.Init();
        b_.Init();
        c_.Init();
    }

    __aicore__ inline void Uninit(MutexBufferManager<bufferType>& bufferManager)
    {
        a_.UnInit();
        b_.UnInit();
        c_.UnInit();
    }

    __aicore__ inline MutexBuffer<bufferType, syncType>& Get()
    {
        if (flag1_ == 0) {
            flag1_ = 1;
            return a_;
        } else if (flag1_ == 1) {
            flag1_ = NUM_2;
            return b_;
        } else {
            flag1_ = 0;
            return c_;
        }
    }

    __aicore__ inline MutexBuffer<bufferType, syncType>& GetVec()
    {
        if (flag1_vec1_ == 0) {
            flag1_vec1_ = 1;
            return a_;
        } else if (flag1_vec1_ == 1) {
            flag1_vec1_ = NUM_2;
            return b_;
        } else {
            flag1_vec1_ = 0;
            return c_;
        }
    }

    __aicore__ inline MutexBuffer<bufferType, syncType>& GetCube()
    {
        if (flag1_bmm2_ == 0) {
            flag1_bmm2_ = 1;
            return a_;
        } else if (flag1_bmm2_ == 1) {
            flag1_bmm2_ = NUM_2;
            return b_;
        } else {
            flag1_bmm2_ = 0;
            return c_;
        }
    }

    __aicore__ inline MutexBuffer<bufferType, syncType>& GetPre()
    {
        if (flag1_ == 0) {
            return c_;
        } else if (flag1_ == 1) {
            return a_;
        } else {
            return b_;
        }
    }

    __aicore__ inline MutexBuffer<bufferType, syncType>& GetReused()
    {
        if (flag2_ == 0) {
            flag2_ = 1;
            return a_;
        } else if (flag2_ == 1) {
            flag2_ = NUM_2;
            return b_;
        } else {
            flag2_ = 0;
            return c_;
        }
    }

    __aicore__ inline MutexBuffer<bufferType, syncType>& GetReused(bool isNextS2IdxNoChange)
    {
        if (isNextS2IdxNoChange) {
            if (flag2_ == 0) {
                return a_;
            } else if (flag2_ == 1) {
                return b_;
            } else {
                return c_;
            }
        } else {
            return GetReused();
        }
    }

private:
    MutexBuffer<bufferType, syncType> a_;
    MutexBuffer<bufferType, syncType> b_;
    MutexBuffer<bufferType, syncType> c_;
    uint32_t flag1_ = 0;
    uint32_t flag1_vec1_ = 0;
    uint32_t flag1_bmm2_ = 0;
    uint32_t flag2_ = 0;
};

template <BufferType bufferType, SyncType syncType = SyncType::INNER_CORE_SYNC>
class MutexBuffersPolicy4buff {
public:
    __aicore__ inline void Init(MutexBufferManager<bufferType>& bufferManager, uint32_t size)
    {
        a_ = detail::AllocMutexBuf<bufferType, syncType>(bufferManager, size);
        b_ = detail::AllocMutexBuf<bufferType, syncType>(bufferManager, size);
        c_ = detail::AllocMutexBuf<bufferType, syncType>(bufferManager, size);
        d_ = detail::AllocMutexBuf<bufferType, syncType>(bufferManager, size);

        a_.Init();
        b_.Init();
        c_.Init();
        d_.Init();
    }

    __aicore__ inline void Uninit(MutexBufferManager<bufferType>& bufferManager)
    {
        a_.UnInit();
        b_.UnInit();
        c_.UnInit();
        d_.UnInit();
    }

    __aicore__ inline MutexBuffer<bufferType, syncType>& Get()
    {
        if (flag1_ == 0) {
            flag1_ = 1;
            return a_;
        } else if (flag1_ == 1) {
            flag1_ = NUM_2;
            return b_;
        } else if (flag1_ == NUM_2) {
            flag1_ = NUM_3;
            return c_;
        } else {
            flag1_ = 0;
            return d_;
        }
    }

    // Q复用
    __aicore__ inline MutexBuffer<bufferType, syncType>& GetPre()
    {
        if (flag1_ == 0) {
            return d_;
        } else if (flag1_ == 1) {
            return a_;
        } else if (flag1_ == NUM_2) {
            return b_;
        } else {
            return c_;
        }
    }

    // KV复用
    __aicore__ inline MutexBuffer<bufferType, syncType>& GetReused()
    {
        if (flag2_ == 0) {
            flag2_ = 1;
            return a_;
        } else if (flag2_ == 1) {
            flag2_ = NUM_2;
            return b_;
        } else if (flag2_ == NUM_2) {
            flag2_ = NUM_3;
            return c_;
        } else {
            flag2_ = 0;
            return d_;
        }
    }

private:
    MutexBuffer<bufferType, syncType> a_;
    MutexBuffer<bufferType, syncType> b_;
    MutexBuffer<bufferType, syncType> c_;
    MutexBuffer<bufferType, syncType> d_;
    uint32_t tail_ = 0;
    uint32_t head_ = 0;
    uint32_t used_ = 0;
    uint32_t flag1_ = 0;
    uint32_t flag2_ = 0;
};

template <BufferType bufferType, SyncType syncType = SyncType::INNER_CORE_SYNC>
class MutexMatrix2x2BufferPolicy {
public:
    __aicore__ inline void Init(MutexBufferManager<bufferType>& bufferManager, uint32_t size)
    {
        bufferM0k0_ = detail::AllocMutexBuf<bufferType, syncType>(bufferManager, size);
        bufferM0k1_ = detail::AllocMutexBuf<bufferType, syncType>(bufferManager, size);
        bufferM1k0_ = detail::AllocMutexBuf<bufferType, syncType>(bufferManager, size);
        bufferM1k1_ = detail::AllocMutexBuf<bufferType, syncType>(bufferManager, size);

        bufferM0k0_.Init();
        bufferM0k1_.Init();
        bufferM1k0_.Init();
        bufferM1k1_.Init();
    }

    __aicore__ inline void Uninit(MutexBufferManager<bufferType>& bufferManager)
    {
        bufferM0k0_.UnInit();
        bufferM0k1_.UnInit();
        bufferM1k0_.UnInit();
        bufferM1k1_.UnInit();
    }

    __aicore__ inline void SetMExtent(int32_t mExtent)
    {
        aIdx_ = -1;
        amIdx_ = (amIdx_ + mSize_ - 1) % mSize_;
        akIdx_ = 0;

        uIdx_ = -1;
        umIdx_ = (umIdx_ + mSize_ - 1) % mSize_;
        ukIdx_ = 0;

        fIdx_ = -1;
        fmIdx_ = (fmIdx_ + mSize_ - 1) % mSize_;
        fkIdx_ = 0;

        mExtent_ = mExtent;
    }

    __aicore__ inline MutexBuffer<bufferType, syncType>& AllocNext()
    {
        aIdx_++;
        return GetBuffer(aIdx_, amIdx_, akIdx_);
    }

    __aicore__ inline MutexBuffer<bufferType, syncType>& ReuseNext()
    {
        uIdx_++;
        return GetBuffer(uIdx_, umIdx_, ukIdx_);
    }

    __aicore__ inline MutexBuffer<bufferType, syncType>& FreeNext()
    {
        fIdx_++;
        return GetBuffer(fIdx_, fmIdx_, fkIdx_);
    }

    __aicore__ inline MutexBuffer<bufferType, syncType>& PeekNextK()
    {
        return PeekBuffer(amIdx_, (1 - akIdx_));
    }

private:
    __aicore__ inline MutexBuffer<bufferType, syncType>& GetBuffer(int32_t xIdx, int32_t& mIdx, int32_t& kIdx)
    {
        mIdx = (mIdx + mExtent_ - 1) % mExtent_;
        kIdx = (xIdx / mExtent_) % kSize_;
        if (mIdx == 0 && kIdx == 0) {
            return bufferM0k0_;
        } else if (mIdx == 0 && kIdx == 1) {
            return bufferM0k1_;
        } else if (mIdx == 1 && kIdx == 0) {
            return bufferM1k0_;
        } else {
            return bufferM1k1_;
        }
    }

    __aicore__ inline MutexBuffer<bufferType, syncType>& PeekBuffer(int32_t mIdx, int32_t kIdx)
    {
        if (mIdx == 0 && kIdx == 0) {
            return bufferM0k0_;
        } else if (mIdx == 0 && kIdx == 1) {
            return bufferM0k1_;
        } else if ((mIdx == 1) && (kIdx == 0)) {
            return bufferM1k0_;
        } else {
            return bufferM1k1_;
        }
    }

    MutexBuffer<bufferType, syncType> bufferM0k0_;
    MutexBuffer<bufferType, syncType> bufferM0k1_;
    MutexBuffer<bufferType, syncType> bufferM1k0_;
    MutexBuffer<bufferType, syncType> bufferM1k1_;
    int32_t mSize_ = NUM_2;
    int32_t kSize_ = NUM_2;

    int32_t aIdx_ = -1;
    int32_t amIdx_ = 0;
    int32_t akIdx_ = 0;

    int32_t uIdx_ = -1;
    int32_t umIdx_ = 0;
    int32_t ukIdx_ = 0;

    int32_t fIdx_ = -1;
    int32_t fmIdx_ = 0;
    int32_t fkIdx_ = 0;

    int32_t mExtent_ = 0;
};
} // namespace fa_base_matmul
#endif
