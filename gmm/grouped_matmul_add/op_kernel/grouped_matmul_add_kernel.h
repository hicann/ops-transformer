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
 * \file grouped_matmul_add_kernel.h
 * \brief
 */
#ifndef __GROUPED_MATMUL_ADD_KERNEL_H__
#define __GROUPED_MATMUL_ADD_KERNEL_H__

#ifdef __CCE_KT_TEST__
#include "stub_def.h"
#include "stub_fun.h"
#endif

#include "kernel_tiling/kernel_tiling.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "lib/matmul_intf.h"

namespace AscendC {

constexpr uint16_t MAX_TENSOR_LIST_SIZE = 128;
constexpr int32_t MKN_LIST_LEN = 128;                                // 128: predefined array legnth
constexpr uint32_t UB_BLOCK_UNIT_SIZE = 32;                          // 32: a block has 32 bytes data
constexpr uint32_t UB_BLOCK_DOUBLE_UNIT_SIZE = 64;                   // 64: a block has 64 bytes data
constexpr uint32_t HALF_UB_BLOCK_UNIT_SIZE = UB_BLOCK_UNIT_SIZE / 2; // 2: a float16 data has two bytes
constexpr uint32_t GROUP_LIST_TYPE_CUMSUM_0 = 0;

template <class AT_, class BT_, class CT_, const MatmulConfig &MM_CFG = CFG_MDL>
struct MmImplType {
    using AT = AT_;
    using BT = BT_;
    using CT = CT_;
    using MT = matmul::Matmul<AT, BT, CT, CT, MM_CFG>;
};

__aicore__ inline uint32_t AlignDown(uint32_t a, uint32_t base)
{
    if (unlikely(base == 0)) {
        return a;
    }
    return a / base * base;
}

#define GET_TILING_DATA_MEMBER_ADDR(tilingType, member, var, tiling) \
    size_t offset##var = (size_t)(&((tilingType *)0)->member); \
    __gm__ uint8_t *(var) = (tiling) + (offset##var)

__aicore__ inline int32_t GetSplitValueFromGroupList(uint32_t groupIdx, int32_t &preOffset, uint32_t groupListType,
                                                     const AscendC::GlobalTensor<int64_t> &groupListGm)
{
    int32_t splitValue = 0;
    AscendC::DataCacheCleanAndInvalid<int64_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(
        groupListGm);
    if (groupListType == GROUP_LIST_TYPE_CUMSUM_0) {
        // gListType = 0: cumulative sum mode
        int32_t offset = static_cast<int32_t>(groupListGm.GetValue(groupIdx));
        splitValue = offset - preOffset;
        preOffset = offset;
    } else {
        // gListType = 1: direct length mode
        splitValue = static_cast<int32_t>(groupListGm.GetValue(groupIdx));
    }
    return splitValue;
}
} // namespace AscendC
#endif // GROUPED_MATMUL_ADD_H
