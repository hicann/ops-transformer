/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file lightning_indexer_v2_base_arch35.h
 * \brief Common arch35 definitions shared by LightningIndexer variants.
 */
#ifndef LIGHTNING_INDEXER_V2_BASE_ARCH35_H
#define LIGHTNING_INDEXER_V2_BASE_ARCH35_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

namespace LIV2Common {

struct LightningIndexerV2SplitCoreInfo {
    uint32_t s2Start = 0U; // S2的起始位置
    uint32_t s2End = 0U;   // S2循环index上限
    uint32_t bN2Start = 0U;
    uint32_t bN2End = 0U;
    uint32_t gS1Start = 0U;
    uint32_t gS1End = 0U;
    bool isLD = false; // 当前核是否需要进行Decode归约任务
    bool isCoreEnable = false;
};

struct LightningIndexerV2LdSplitCoreInfo {
    bool isLdCoreEnable = false;    // 当前核是否参与规约任务
    uint32_t saveWorkSpaceIdx = 0U; // 存放LD参数的地址
    uint32_t bn2Idx = 0U;           // 归约任务
    uint32_t bIdx = 0U;
    uint32_t n2Idx = 0U;
    uint32_t mIdx = 0U;
    uint32_t workspaceIdx = 0U; // 当前AIV核上规约任务的索引
    uint32_t workspaceNum = 0U; // 当前AIV核上规约任务的S2切分数量
    uint32_t mStart = 0U;
    uint32_t mNum = 0U;
    uint64_t indiceOutCoreOffset = 0ULL; // 最终输出索引搬出Topk的初始偏移地址
};

template <typename T>
__aicore__ inline T Align(T num, T rnd)
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd) * (rnd)));
}

template <typename T1, typename T2>
__aicore__ inline T1 Min(T1 a, T2 b)
{
    return (a > b) ? (b) : (a);
}

template <typename T1, typename T2>
__aicore__ inline T1 Max(T1 a, T2 b)
{
    return (a > b) ? (a) : (b);
}

template <typename T>
__aicore__ inline T CeilDiv(T num, T rnd)
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd)));
}

} // namespace LIV2Common

// bank冲突优化
// david 256KB bank layout
// shape  (             bank_depth  (            banks  bank_groups  block))  (512  (  2   8  32))
// stride (banks*bank_groups*block  (bank_groups*block        block      1))  (512  (256  32   1))
#define UB_BLOCK 32 // 32B
#define UB_BANK_GROUPS 8
#define UB_BANKS 2
#define UB_BANK_DEPTH 512

#define UB_BANK_GROUP_STRIDE UB_BLOCK                               // 32B
#define UB_BANK_STRIDE (UB_BANK_GROUPS * UB_BLOCK)                  // 256B
#define UB_BANK_DEPTH_STRIDE (UB_BANKS * UB_BANK_GROUPS * UB_BLOCK) // 512B

#endif // LIGHTNING_INDEXER_V2_BASE_ARCH35_H
