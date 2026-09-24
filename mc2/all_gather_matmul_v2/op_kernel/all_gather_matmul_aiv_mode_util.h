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
#include "../../common/op_kernel/mc2_aiv_data_copy_common.h"
#include "../../common/op_kernel/mc2_aiv_sync_common.h"
#include "../../common/op_kernel/mc2_matmul_aiv_kernel_common.h"

using namespace AscendC;

#define FORCE_INLINE_AICORE __attribute__((always_inline)) inline __aicore__

constexpr static int32_t AIC_WAIT_AIV_FINISH_ALIGN_FLAG_ID = 12;
constexpr static size_t BLOCK_32_BYTES = 32U;
constexpr static size_t BLOCK_256_BYTES = 256U;
constexpr static size_t BLOCK_512_BYTES = 512U;
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
using Block32B = BaseBlock<T, BLOCK_32_BYTES>;

template <typename T>
using Block256B = BaseBlock<T, BLOCK_256_BYTES>;

template <typename T>
using Block512B = BaseBlock<T, BLOCK_512_BYTES>;

FORCE_INLINE_AICORE void CheckBuffFlag(__gm__ int32_t *buff, TBuf<AscendC::TPosition::VECCALC> uBuf_, int32_t flag)
{
    SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    LocalTensor<int32_t> ubTensor = uBuf_.AllocTensor<int32_t>();
    while (true) {
        Mc2AivDataCopy::CopyGmToUbufAlignB16(ubTensor, buff, 1, sizeof(int32_t), 0, 0);
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
    Mc2AivDataCopy::CopyUbufToGmAlignB16(buff, ubTensor, 1, sizeof(int32_t), 0, 0);
    uBuf_.FreeTensor<int32_t>(ubTensor);
}

#endif //__ALL_GATHER_MATMUL_AIV_MODE_UTIL_H__
