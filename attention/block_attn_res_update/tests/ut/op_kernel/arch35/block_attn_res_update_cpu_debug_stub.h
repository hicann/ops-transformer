/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BLOCK_ATTN_RES_UPDATE_CPU_DEBUG_STUB_H
#define BLOCK_ATTN_RES_UPDATE_CPU_DEBUG_STUB_H

#include <cstdint>

#include "tikicpulib.h"
#include "blaze_kernel_stub.h"

// CANN 9.2 tikicpulib declares the legacy eight-argument intrinsic, while the
// Ascend950 Tensor API calls the current nine-argument form.
inline void copy_gm_to_cbuf_v2(__cbuf__ void *dst, __gm__ void *src, uint8_t sid, uint32_t nBurst, uint32_t lenBurst,
                               uint8_t padFuncMode, uint64_t l2CacheCtl, uint64_t srcStride, uint32_t dstStride)
{
    (void)l2CacheCtl;
    copy_gm_to_cbuf_v2(dst, src, sid, nBurst, lenBurst, padFuncMode, srcStride, dstStride);
}

#include "kernel_operator.h"
#include "tensor_api/tensor.h"

namespace BlockAttnResUpdateOps::CpuDebug {

template <typename T>
__ubuf__ T *ToPhysicalUbAddress(uint64_t byteOffset)
{
    const uint32_t offset = static_cast<uint32_t>(byteOffset);
    AscendC::LocalTensor<uint8_t> localTensor(AscendC::TPosition::VECCALC, offset, AscendC::TOTAL_UB_SIZE - offset);
    return reinterpret_cast<__ubuf__ T *>(localTensor.GetPhyAddr());
}

} // namespace BlockAttnResUpdateOps::CpuDebug

namespace AscendC::Te {

template <typename PtrPattern, typename DataType, typename Addr, EnableMakePtrByTrait<PtrPattern, Addr> = 0>
__aicore__ inline auto BlockAttnResUpdateMakeMemPtr(Addr byteOffset)
{
    auto *physicalAddress = BlockAttnResUpdateOps::CpuDebug::ToPhysicalUbAddress<DataType>(byteOffset);
    return MakeMemPtr<PtrPattern>(physicalAddress);
}

template <typename PtrPattern, typename Iterator, EnableMakeHardwarePtr<PtrPattern, Iterator> = 0>
__aicore__ inline constexpr auto BlockAttnResUpdateMakeMemPtr(Iterator iterator)
{
    return MakeMemPtr<PtrPattern>(iterator);
}

} // namespace AscendC::Te

// The production kernel keeps using logical UB byte offsets. Redirect only this
// forced-include UT translation unit to the physical-address adapter above.
#define MakeMemPtr BlockAttnResUpdateMakeMemPtr

#endif // BLOCK_ATTN_RES_UPDATE_CPU_DEBUG_STUB_H
