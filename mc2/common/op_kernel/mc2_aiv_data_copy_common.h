/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MC2_AIV_DATA_COPY_COMMON_H
#define MC2_AIV_DATA_COPY_COMMON_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

namespace Mc2AivDataCopy {

using namespace AscendC;

template <typename T>
__attribute__((always_inline)) inline __aicore__ void CopyUbufToGm(__gm__ T *dst, LocalTensor<T> ubTensor,
                                                                   uint16_t nBurst, uint16_t lenBurst,
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
__attribute__((always_inline)) inline __aicore__ void CopyGmToUbuf(LocalTensor<T> ubTensor, __gm__ T *src,
                                                                   uint16_t nBurst, uint32_t lenBurst,
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
__attribute__((always_inline)) inline __aicore__ void CopyGmToUbufAlignB16(LocalTensor<T> ubTensor, __gm__ T *src,
                                                                           uint16_t nBurst, uint32_t lenBurst,
                                                                           uint16_t srcStride, uint16_t dstStride)
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
__attribute__((always_inline)) inline __aicore__ void CopyUbufToGmAlignB16(__gm__ T *dst, LocalTensor<T> ubTensor,
                                                                           uint16_t nBurst, uint32_t lenBurst,
                                                                           uint16_t srcStride, uint16_t dstStride)
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

} // namespace Mc2AivDataCopy

#endif // MC2_AIV_DATA_COPY_COMMON_H
