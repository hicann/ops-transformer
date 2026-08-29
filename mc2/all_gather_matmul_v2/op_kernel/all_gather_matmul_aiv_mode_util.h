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
 * \file all_gather_matmul_v2_util.h
 * \brief
 */

#ifndef __ALL_GATHER_MATMUL_AIV_MODE_UTIL_H__
#define __ALL_GATHER_MATMUL_AIV_MODE_UTIL_H__

#pragma once
#include "../../3rd/template_linear_algebra/op_kernel/template_linear_algebra/tla_numeric_size.hpp"

using namespace AscendC;

#define FORCE_INLINE_AICORE __attribute__((always_inline)) inline __aicore__

constexpr static int32_t AIC_WAIT_AIV_FINISH_ALIGN_FLAG_ID = 12;
template <typename T, size_t SIZE>
struct BaseBlock {
    static_assert((SIZE & (SIZE - 1)) == 0, "Invalid block size");
    static constexpr size_t size = Catlass::BytesToBits(SIZE) / Catlass::SizeOfBits<T>::value;

    static FORCE_INLINE_AICORE size_t Count(size_t len)
    {
        return (len + size - 1) / size;
    }

    static FORCE_INLINE_AICORE bool IsAligned(size_t len)
    {
        return len % size == 0;
    }

    static FORCE_INLINE_AICORE size_t AlignUp(size_t len)
    {
        return (len + size - 1) & ~(size - 1);
    }

    static FORCE_INLINE_AICORE size_t AlignDown(size_t len)
    {
        return len & ~(size - 1);
    }
};

template <typename T>
using Block32B = BaseBlock<T, 32>;

template <typename T>
using Block256B = BaseBlock<T, 256>;

template <typename T>
using Block512B = BaseBlock<T, 512>;

inline __aicore__ void AlignJudge(bool trans_a, bool trans_b, int32_t m, int32_t k, int32_t n, int32_t m_align,
                                  int32_t k_align, int32_t n_align, int32_t &aligned_a, int32_t &aligned_b)
{
    if (!trans_a) {
        aligned_a = k != k_align;
    } else {
        aligned_a = (m != m_align && m != 1);
    }

    if (!trans_b) {
        aligned_b = (n != n_align);
    } else {
        aligned_b = (k != k_align);
    }
}

template <pipe_t pipe, uint64_t mode>
inline __aicore__ void FFTSCrossCoreSync(uint64_t flag_id)
{
    AscendC::CrossCoreSetFlag<mode, pipe>(flag_id);
}
__aicore__ inline void SetAndWaitAivSync(uint64_t flag_idx, int32_t pipe_depth = 2)
{
    FFTSCrossCoreSync<PIPE_MTE3, 0>(flag_idx + pipe_depth);
    WaitEvent(flag_idx + pipe_depth);
}
__aicore__ inline void SetAicSync(uint64_t flag_idx)
{
    FFTSCrossCoreSync<PIPE_MTE3, 2>(flag_idx);
}

template <typename T>
FORCE_INLINE_AICORE void CopyUbufToGm(__gm__ T *dst, LocalTensor<T> ubTensor, uint16_t nBurst, uint16_t lenBurst,
                                      uint16_t srcStride, uint16_t dstStride)
{
    DataCopyParams dataCopyParams(nBurst,    // blockCount
                                  lenBurst,  // blockLen
                                  srcStride, // srcStride
                                  dstStride  // dstStride
    );
    GlobalTensor<T> gmTensor;
    gmTensor.SetGlobalBuffer(dst);
    DataCopy(gmTensor, ubTensor, dataCopyParams);
}

template <typename T>
FORCE_INLINE_AICORE void CopyGmToUbuf(LocalTensor<T> ubTensor, __gm__ T *src, uint16_t nBurst, uint32_t lenBurst,
                                      uint16_t srcStride, uint16_t dstStride)
{
    DataCopyParams dataCopyParams(nBurst,    // blockCount
                                  lenBurst,  // blockLen
                                  srcStride, // srcStride
                                  dstStride  // dstStride
    );
    GlobalTensor<T> gmTensor;
    gmTensor.SetGlobalBuffer(src);
    DataCopy(ubTensor, gmTensor, dataCopyParams);
}

template <typename T>
FORCE_INLINE_AICORE void CopyGmToUbufAlignB16(LocalTensor<T> ubTensor, __gm__ T *src, uint16_t nBurst,
                                              uint32_t lenBurst, uint16_t srcStride, uint16_t dstStride)
{
    DataCopyExtParams dataCopyParams(nBurst,    // blockCount
                                     lenBurst,  // blockLen
                                     srcStride, // srcStride
                                     dstStride, // dstStride
                                     0);
    GlobalTensor<T> gmTensor;
    gmTensor.SetGlobalBuffer(src);
    DataCopyPadExtParams<T> padParams;
    DataCopyPad(ubTensor, gmTensor, dataCopyParams, padParams);
}

template <typename T>
FORCE_INLINE_AICORE void CopyUbufToGmAlignB16(__gm__ T *dst, LocalTensor<T> ubTensor, uint16_t nBurst,
                                              uint32_t lenBurst, uint16_t srcStride, uint16_t dstStride)
{
    DataCopyExtParams dataCopyParams(nBurst,    // blockCount
                                     lenBurst,  // blockLen
                                     srcStride, // srcStride
                                     dstStride, // dstStride
                                     0);
    GlobalTensor<T> gmTensor;
    gmTensor.SetGlobalBuffer(dst);
    DataCopyPad(gmTensor, ubTensor, dataCopyParams);
}

FORCE_INLINE_AICORE void CheckBuffFlag(__gm__ int32_t *buff, TBuf<AscendC::TPosition::VECCALC> uBuf_, int32_t flag)
{
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    LocalTensor<int32_t> ubTensor = uBuf_.AllocTensor<int32_t>();
    while (true) {
        CopyGmToUbufAlignB16(ubTensor, buff, 1, sizeof(int32_t), 0, 0);
        SetFlag<HardEvent::MTE2_S>(EVENT_ID3);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID3); // Scalar等MTE2
        if (ubTensor(0) == flag) {
            break;
        }
    }
    uBuf_.FreeTensor<int32_t>(ubTensor);
}

FORCE_INLINE_AICORE void SetBuffFlag(__gm__ int32_t *buff, TBuf<AscendC::TPosition::VECCALC> uBuf_, int32_t flag)
{
    SetFlag<HardEvent::MTE3_S>(EVENT_ID2);
    WaitFlag<HardEvent::MTE3_S>(EVENT_ID2);
    LocalTensor<int32_t> ubTensor = uBuf_.AllocTensor<int32_t>();
    ubTensor(0) = flag;
    SetFlag<HardEvent::S_MTE3>(EVENT_ID2);
    WaitFlag<HardEvent::S_MTE3>(EVENT_ID2);
    CopyUbufToGmAlignB16(buff, ubTensor, 1, sizeof(int32_t), 0, 0);
    uBuf_.FreeTensor<int32_t>(ubTensor);
}

FORCE_INLINE_AICORE void SetBuffFlagByAdd(__gm__ int32_t *buff, TBuf<AscendC::TPosition::VECCALC> uBuf_, int32_t flag)
{
    PipeBarrier<PIPE_ALL>();
    LocalTensor<int32_t> ubTensor = uBuf_.AllocTensor<int32_t>();
    ubTensor(0) = flag;
    PipeBarrier<PIPE_ALL>();
    SetAtomicAdd<int32_t>();
    PipeBarrier<PIPE_ALL>();
    CopyUbufToGmAlignB16(buff, ubTensor, 1, sizeof(int32_t), 0, 0);
    PipeBarrier<PIPE_ALL>();
    SetAtomicNone();
    PipeBarrier<PIPE_ALL>();
}

inline __aicore__ void GetBlockIdx(int32_t loop_idx, int32_t m_loop, int32_t n_loop, int32_t swizzl_direction,
                                   int32_t swizzl_count, int64_t &m_idx, int64_t &n_idx)
{
    uint32_t in_batch_idx = loop_idx % (m_loop * n_loop);
    if (swizzl_direction == 0) { // Zn
        uint32_t tile_block_loop = (m_loop + swizzl_count - 1) / swizzl_count;
        uint32_t tile_block_idx = in_batch_idx / (swizzl_count * n_loop);
        uint32_t in_tile_block_idx = in_batch_idx % (swizzl_count * n_loop);

        uint32_t n_row = swizzl_count;
        if (tile_block_idx == tile_block_loop - 1) {
            n_row = m_loop - swizzl_count * tile_block_idx;
        }
        m_idx = tile_block_idx * swizzl_count + in_tile_block_idx % n_row;
        n_idx = in_tile_block_idx / n_row;
        if (tile_block_idx % 2 != 0) {
            n_idx = n_loop - n_idx - 1;
        }
    } else if (swizzl_direction == 1) { // Nz
        uint32_t tile_block_loop = (n_loop + swizzl_count - 1) / swizzl_count;
        uint32_t tile_block_idx = in_batch_idx / (swizzl_count * m_loop);
        uint32_t in_tile_block_idx = in_batch_idx % (swizzl_count * m_loop);

        uint32_t n_col = swizzl_count;
        if (tile_block_idx == tile_block_loop - 1) {
            n_col = n_loop - swizzl_count * tile_block_idx;
        }
        m_idx = in_tile_block_idx / n_col;
        n_idx = tile_block_idx * swizzl_count + in_tile_block_idx % n_col;
        if (tile_block_idx % 2 != 0) {
            m_idx = m_loop - m_idx - 1;
        }
    }
}

#endif //__ALL_GATHER_MATMUL_AIV_MODE_UTIL_H__
