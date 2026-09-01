/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef VF_SOFTMAX_ALL_INVALID_H_
#define VF_SOFTMAX_ALL_INVALID_H_
#include "kernel_tensor.h"
#include "vf_common_def.h"
namespace NpuArch::Epilogue::Block::Mxfp4VF {
using AscendC::LocalTensor;
using namespace AscendC;
using namespace MicroAPI;

// 通用 chunk 全0 模板：pDest 清 0 + local_group_max 填 MIN_VALUE + global_max 置极负
template <typename T, bool HAS_HIGH_OFF = true, uint16_t GROUPS = 4, uint16_t CHUNKS = 2, uint16_t LGM_GROUPS = 0,
          uint16_t CHUNK_START = 0>
__simd_vf__ inline void softmax_all_invalid_chunk_vf(__ubuf__ uint8_t *p_dest, __ubuf__ T *local_group_max)
{
    constexpr uint16_t P_GROUP_STRIDE = 2048; // 每 32 行组 P 字节数(qs128/qs64 均为 2048)
    constexpr uint16_t P_HIGH_OFF = 16256;    // qs128 高位 128 字节远端地址
    constexpr uint16_t P_HIGH_STEP = 16384;   // qs128 高位步进
    constexpr uint16_t LGM_GROUP = 128;       // 每 32 行组 local_group_max 元素数(128/64 布局均 128)

    MaskReg preg_all_16 = CreateMask<uint16_t, MaskPattern::ALL>();
    MaskReg preg_all_8 = CreateMask<uint8_t, MaskPattern::ALL>();
    MaskReg preg_vl128 = CreateMask<uint8_t, MaskPattern::VL128>();
    MaskReg preg_vl128_not;
    MaskNot(preg_vl128_not, preg_vl128, preg_all_8);

    RegTensor<uint8_t> zero8;
    RegTensor<T> min_reg;
    Duplicate(zero8, static_cast<uint8_t>(0));
    Duplicate(min_reg, MIN_VALUE);

    // pDest 全 0（组循环 + 高低位写，与主 VF padding 分支一致）
    for (uint16_t c = CHUNK_START; c < CHUNK_START + CHUNKS; ++c) {
        const uint16_t groupBase = c * GROUPS;
        for (uint16_t g = 0; g < GROUPS; ++g) {
            for (uint16_t j = 0; j < 8; j += 2) {
                StoreAlign(((__ubuf__ uint8_t *&)p_dest) + (groupBase + g) * P_GROUP_STRIDE + j * 256, zero8,
                           preg_vl128);
                if constexpr (HAS_HIGH_OFF) {
                    StoreAlign(((__ubuf__ uint8_t *&)p_dest) + (groupBase + g) * P_GROUP_STRIDE + j * 256 + P_HIGH_OFF,
                               zero8, preg_vl128_not);
                }
                StoreAlign(((__ubuf__ uint8_t *&)p_dest) + (groupBase + g) * P_GROUP_STRIDE + j * 256 + 128, zero8,
                           preg_vl128);
                if constexpr (HAS_HIGH_OFF) {
                    StoreAlign(((__ubuf__ uint8_t *&)p_dest) + (groupBase + g) * P_GROUP_STRIDE + j * 256 + P_HIGH_STEP,
                               zero8, preg_vl128_not);
                }
            }
        }
    }
    // local_group_max 填 MIN_VALUE（LGM_GROUPS 组 × LGM_GROUP; 默认 CHUNKS*GROUPS 组）
    constexpr uint16_t lgmGroups = (LGM_GROUPS == 0) ? CHUNKS * GROUPS : LGM_GROUPS;
    constexpr uint16_t lgmBase = CHUNK_START * GROUPS * LGM_GROUP;
    for (uint16_t i = lgmBase; i < lgmBase + lgmGroups * LGM_GROUP; i += 128) {
        StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B16>(local_group_max + i, min_reg, preg_all_16);
    }
    // 注意: 这里**不写** global_max —— 见本函数上方注释
}

template <typename T, bool HAS_HIGH_OFF = true, uint16_t GROUPS = 4, uint16_t CHUNKS = 2, uint16_t LGM_GROUPS = 0,
          uint16_t CHUNK_START = 0>
__aicore__ inline void SoftmaxAllInvalidCallVF(const LocalTensor<uint8_t> &pDest, const LocalTensor<T> &localGroupMax)
{
    softmax_all_invalid_chunk_vf<T, HAS_HIGH_OFF, GROUPS, CHUNKS, LGM_GROUPS, CHUNK_START>(
        (__ubuf__ uint8_t *)pDest.GetPhyAddr(), (__ubuf__ T *)localGroupMax.GetPhyAddr());
}

template <typename T>
__simd_vf__ inline void init_global_max_vf(__ubuf__ T *global_max)
{
    MaskReg preg_all_16 = CreateMask<uint16_t, MaskPattern::ALL>();
    RegTensor<T> neg_reg;
    Duplicate(neg_reg, static_cast<T>(-60000));
    StoreAlign<T, MicroAPI::StoreDist::DIST_NORM_B16>(global_max, neg_reg, preg_all_16);
}

template <typename T>
__aicore__ inline void InitGlobalMaxCallVF(const LocalTensor<T> &localGlobalMax)
{
    init_global_max_vf<T>((__ubuf__ T *)localGlobalMax.GetPhyAddr());
}

} // namespace NpuArch::Epilogue::Block::Mxfp4VF
#endif // VF_SOFTMAX_ALL_INVALID_H_
