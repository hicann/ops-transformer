/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * \file sort_lib_params.h
 * \brief SortLib kernel-side parameter struct.
 */

#ifndef SORT_LIB_PARAMS_H
#define SORT_LIB_PARAMS_H

#include <cstdint>

namespace SortLib {

struct SortParams {
    uint32_t numTileData = 0;  // 每 tile 最大元素数（受 UB 容量限制，由 ComputeTileData 求得）
    uint32_t tileCount = 0;    // 总 tile 数 = ceil(totalElements / numTileData)
    uint32_t activeCores = 0;  // 实际使用的 AI Core 数 = min(totalJobs, tileCount)
    uint32_t tmpUbSize = 0;    // AscendC::Sort 临时 UB 空间大小（字节）
    int64_t totalElements = 0; // 待排序元素总数
    uint32_t isSingleCore = 0; // 单核/多核标志：1=单核快路径，0=多核 LSD radix sort
    uint32_t jobIndex = 0;     // 本核的排序 job 编号([0, totalJobs) 的连续分片编号;
                               // 调用方负责从物理核号/角色翻译, 模块不感知核号)
    uint32_t totalJobs = 0;    // 参与排序的 job 总数(= 物理参与核数, 全核模式传核数)
    uint32_t radixRounds = 0;  // radix 轮数(每轮按 key 的一个字节 LSD 排序): 0=默认 sizeof(ValT);
                               // key 高位字节为常量(如非负且值域 < 2^24 的 int32)时, 调用方可传
                               // ceil(值域位宽/8) 裁掉无效轮, 每轮省 2 次核间屏障与一轮扫描;
                               // workspace 仍按 sizeof(ValT) 轮布局(与 host tiling 契约一致), 仅循环缩短
};

} // namespace SortLib

#endif
