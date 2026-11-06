/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_COPY_H
#define MEGA_MOE_COPY_H

#include <type_traits>
#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 9201) || (__NPU_ARCH__ == 9202))
#include "mega_moe_arch_policy.h"
#else
#include "c_api/asc_simd.h"
#endif

namespace MegaMoeImpl {

#if defined(MEGA_MOE_ARCH92_DEVICE)
enum class GmLoadCacheMode : uint8_t {
    NORMAL_FIRST_VICTIM = 0,
    NORMAL_LAST_VICTIM = 1,
    NORMAL_PERSISTENT = 2,
    NOTALLOC_KEEP = 4,
    NOTALLOC_CLEAN = 5,
    NOTALLOC_DROP = 6
};

enum class GmStoreCacheMode : uint8_t {
    NORMAL_FIRST_VICTIM = 0,
    NORMAL_LAST_VICTIM = 1,
    NORMAL_PERSISTENT = 2,
    NOTALLOC_CLEAN = 4
};
#else
using GmLoadCacheMode = asc_load_l2_cache_mode;
using GmStoreCacheMode = asc_store_l2_cache_mode;
#endif

// All lengths/pitches are bytes. Pitches are row-start to row-start distances.
// UB pitch == rowBytes selects compact packing; otherwise UB pitch must be 32B aligned.
struct CopyRows {
    uint16_t rows;
    uint32_t rowBytes;
    uint64_t gmPitchBytes;
    uint32_t ubPitchBytes;
};

// C API configures padding once; Basic API supplies it with each padded transfer.
__aicore__ inline void InitCopyPadding()
{
#if !defined(MEGA_MOE_ARCH92_DEVICE)
    asc_set_copy_pad_val(uint32_t{0});
#endif
}

// Keep the source element width for the hardware's automatic dummy padding.
// The instruction handles 64-bit elements as 32-bit words.
template <typename T>
using CopyWord = std::conditional_t<sizeof(T) == 1, uint8_t, std::conditional_t<sizeof(T) == 2, uint16_t, uint32_t>>;

#if defined(MEGA_MOE_ARCH92_DEVICE)
namespace Arch92Copy {

// These are borrowed views of existing storage, with no allocation or persistent state.
template <typename T>
__aicore__ inline AscendC::LocalTensor<T> UbView(__ubuf__ T* address, uint32_t bytes)
{
    AscendC::TBuffAddr buffer{};
    buffer.bufferAddr = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(address));
    buffer.dataLen = bytes;
    buffer.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECCALC);
    AscendC::LocalTensor<T> view;
    view.SetAddr(buffer);
    return view;
}

template <typename T>
__aicore__ inline AscendC::GlobalTensor<T> GmView(__gm__ T* address, uint8_t cacheMode, CopyRows copy)
{
    // Basic API carries the instruction's cache hint in the GM address tag.
    auto tagged = (reinterpret_cast<uint64_t>(address) & AscendC::L2_CACHE_OFFSET_MASK) |
                  (static_cast<uint64_t>(cacheMode) << AscendC::L2_CACHE_OFFSET);
    AscendC::GlobalTensor<T> view;
    const uint64_t bytes = copy.rows == 0 ? 0 : (copy.rows - 1U) * copy.gmPitchBytes + copy.rowBytes;
    view.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(tagged), (bytes + sizeof(T) - 1U) / sizeof(T));
    return view;
}

constexpr uint32_t BLOCK_BYTES = 32U;

__aicore__ inline uint32_t AlignedRowBytes(uint32_t bytes)
{
    return (bytes + BLOCK_BYTES - 1U) / BLOCK_BYTES * BLOCK_BYTES;
}

__aicore__ inline uint32_t UbSpan(CopyRows copy)
{
    return copy.rows == 0 ? 0 : (copy.rows - 1U) * copy.ubPitchBytes + AlignedRowBytes(copy.rowBytes);
}

} // namespace Arch92Copy
#endif

// These functions enqueue a transfer only. The caller owns events and buffer reuse.
// C API callers initialize the constant pad value before enabling constant padding.
template <typename D, typename S>
__aicore__ inline void CopyGmToUb(__ubuf__ D* dst, __gm__ S* src, CopyRows copy,
                                  GmLoadCacheMode cacheMode = GmLoadCacheMode::NORMAL_FIRST_VICTIM,
                                  bool enableConstantPad = false)
{
#if defined(MEGA_MOE_ARCH92_DEVICE)
    using Word = CopyWord<S>;
    auto local = Arch92Copy::UbView(reinterpret_cast<__ubuf__ Word*>(dst), Arch92Copy::UbSpan(copy));
    auto global = Arch92Copy::GmView(reinterpret_cast<__gm__ Word*>(src), static_cast<uint8_t>(cacheMode), copy);
    const int64_t gmGap = copy.rows > 1U ? static_cast<int64_t>(copy.gmPitchBytes) - copy.rowBytes : 0;
    AscendC::DataCopyPadExtParams<Word> pad{enableConstantPad, 0, 0, Word{0}};
    if (copy.ubPitchBytes == copy.rowBytes) {
        AscendC::DataCopyPad<Word, AscendC::PaddingMode::Compact>(
            local, global, AscendC::DataCopyExtParams{copy.rows, copy.rowBytes, gmGap, 0, 0}, pad);
    } else {
        const int64_t ubGap =
            copy.rows > 1U ? (static_cast<int64_t>(copy.ubPitchBytes) - Arch92Copy::AlignedRowBytes(copy.rowBytes)) /
                                 Arch92Copy::BLOCK_BYTES :
                             0;
        AscendC::DataCopyPad(local, global, AscendC::DataCopyExtParams{copy.rows, copy.rowBytes, gmGap, ubGap, 0}, pad);
    }
#else
    asc_copy_gm2ub_align(reinterpret_cast<__ubuf__ CopyWord<S>*>(dst), reinterpret_cast<__gm__ CopyWord<S>*>(src),
                         copy.rows, copy.rowBytes, 0, 0, enableConstantPad, cacheMode, copy.gmPitchBytes,
                         copy.ubPitchBytes);
#endif
}

template <typename D, typename S>
__aicore__ inline void CopyGmToUb(__ubuf__ D* dst, __gm__ S* src, uint32_t bytes,
                                  GmLoadCacheMode cacheMode = GmLoadCacheMode::NORMAL_FIRST_VICTIM,
                                  bool enableConstantPad = false)
{
    CopyGmToUb(dst, src, CopyRows{1, bytes, 0, 0}, cacheMode, enableConstantPad);
}

template <typename D, typename S>
__aicore__ inline void CopyUbToGm(__gm__ D* dst, __ubuf__ S* src, CopyRows copy,
                                  GmStoreCacheMode cacheMode = GmStoreCacheMode::NORMAL_FIRST_VICTIM)
{
#if defined(MEGA_MOE_ARCH92_DEVICE)
    auto local = Arch92Copy::UbView(reinterpret_cast<__ubuf__ uint8_t*>(src), Arch92Copy::UbSpan(copy));
    auto global = Arch92Copy::GmView(reinterpret_cast<__gm__ uint8_t*>(dst), static_cast<uint8_t>(cacheMode), copy);
    const int64_t gmGap = copy.rows > 1U ? static_cast<int64_t>(copy.gmPitchBytes) - copy.rowBytes : 0;
    if (copy.ubPitchBytes == copy.rowBytes) {
        AscendC::DataCopyPad<uint8_t, AscendC::PaddingMode::Compact>(
            global, local, AscendC::DataCopyExtParams{copy.rows, copy.rowBytes, 0, gmGap, 0});
    } else {
        const int64_t ubGap =
            copy.rows > 1U ? (static_cast<int64_t>(copy.ubPitchBytes) - Arch92Copy::AlignedRowBytes(copy.rowBytes)) /
                                 Arch92Copy::BLOCK_BYTES :
                             0;
        AscendC::DataCopyPad(global, local, AscendC::DataCopyExtParams{copy.rows, copy.rowBytes, ubGap, gmGap, 0});
    }
#else
    asc_copy_ub2gm_align(reinterpret_cast<__gm__ uint8_t*>(dst), reinterpret_cast<__ubuf__ uint8_t*>(src), copy.rows,
                         copy.rowBytes, cacheMode, copy.gmPitchBytes, copy.ubPitchBytes);
#endif
}

template <typename D, typename S>
__aicore__ inline void CopyUbToGm(__gm__ D* dst, __ubuf__ S* src, uint32_t bytes,
                                  GmStoreCacheMode cacheMode = GmStoreCacheMode::NORMAL_FIRST_VICTIM)
{
    CopyUbToGm(dst, src, CopyRows{1, bytes, 0, 0}, cacheMode);
}

// UB-to-UB copies use PIPE_V and require an integral number of 32B blocks.
template <typename D, typename S>
__aicore__ inline void CopyUbToUb(__ubuf__ D* dst, __ubuf__ S* src, uint32_t bytes)
{
#if defined(MEGA_MOE_ARCH92_DEVICE)
    auto output = Arch92Copy::UbView(reinterpret_cast<__ubuf__ uint8_t*>(dst), bytes);
    auto input = Arch92Copy::UbView(reinterpret_cast<__ubuf__ uint8_t*>(src), bytes);
    AscendC::DataCopy(output, input, bytes);
#else
    asc_copy_ub2ub(dst, src, bytes);
#endif
}

} // namespace MegaMoeImpl
#endif
