/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef VF_SOFTMAX_ALL_INVALID_MXFP8_H_
#define VF_SOFTMAX_ALL_INVALID_MXFP8_H_
#include "../mxfp8_vf/vf_common_def_mxfp8.h"

namespace NpuArch::Epilogue::Block::Mxfp8VF {
using AscendC::LocalTensor;
using namespace AscendC;
using namespace MicroAPI;

// 全无效段：P 清 0 + local_group_max 填 MIN；故意不写 global_max（LSE/online max 不受空块推动）。
// HAS_QS128_HI：qs128 额外清 +16384/+24576 高半面。
template <typename T, bool HAS_QS128_HI = true, uint16_t GROUPS = 4, uint16_t GROUP_BASE = 0>
__simd_vf__ inline void softmax_all_invalid_chunk_vf(__ubuf__ uint8_t* p_dest, __ubuf__ T* local_group_max)
{
    constexpr uint16_t P_GROUP_STRIDE = 2048;
    MaskReg preg_all_16 = CreateMask<uint16_t, MaskPattern::ALL>();
    MaskReg preg_vl128 = CreateMask<uint8_t, MaskPattern::VL128>();

    RegTensor<uint8_t> zero8;
    RegTensor<T> min_reg;
    Duplicate(zero8, static_cast<uint8_t>(0));
    Duplicate(min_reg, MIN_VALUE);

    // 与 ZeroPUBUnwrittenKvsGroups 一致：按 256B 槽清 pOff / +8192（及 qs128 的 +16384/+24576）
    for (uint16_t g = 0; g < GROUPS; ++g) {
        const uint16_t gi = GROUP_BASE + g;
        for (uint16_t j = 0; j < 8; j += 2) {
            uint32_t pOff = gi * P_GROUP_STRIDE + j * 256;
            StoreAlign(p_dest + pOff, zero8, preg_vl128);
            StoreAlign(p_dest + pOff + 128, zero8, preg_vl128);
            StoreAlign(p_dest + pOff + 8192, zero8, preg_vl128);
            StoreAlign(p_dest + pOff + 8192 + 128, zero8, preg_vl128);
            if constexpr (HAS_QS128_HI) {
                StoreAlign(p_dest + pOff + 16384, zero8, preg_vl128);
                StoreAlign(p_dest + pOff + 16384 + 128, zero8, preg_vl128);
                StoreAlign(p_dest + pOff + 24576, zero8, preg_vl128);
                StoreAlign(p_dest + pOff + 24576 + 128, zero8, preg_vl128);
            }
        }
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B16>(local_group_max + gi * 128, min_reg, preg_all_16);
    }
}

template <typename T, bool HAS_QS128_HI = true, uint16_t GROUPS = 4, uint16_t GROUP_BASE = 0>
__aicore__ inline void SoftmaxAllInvalidCallVF(const LocalTensor<uint8_t>& pDest, const LocalTensor<T>& localGroupMax)
{
    softmax_all_invalid_chunk_vf<T, HAS_QS128_HI, GROUPS, GROUP_BASE>((__ubuf__ uint8_t*)pDest.GetPhyAddr(),
                                                                      (__ubuf__ T*)localGroupMax.GetPhyAddr());
}

template <typename T>
__simd_vf__ inline void init_global_max_vf(__ubuf__ T* global_max)
{
    MaskReg preg_all_16 = CreateMask<uint16_t, MaskPattern::ALL>();
    RegTensor<T> min_reg;
    Duplicate(min_reg, MIN_VALUE);
    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B16>(global_max, min_reg, preg_all_16);
}

template <typename T>
__aicore__ inline void InitGlobalMaxCallVF(const LocalTensor<T>& globalMax)
{
    init_global_max_vf<T>((__ubuf__ T*)globalMax.GetPhyAddr());
}

} // namespace NpuArch::Epilogue::Block::Mxfp8VF
#endif // VF_SOFTMAX_ALL_INVALID_MXFP8_H_
