/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __MC2_CALC_NUM_BLOCKS_H__
#define __MC2_CALC_NUM_BLOCKS_H__
#include <algorithm>
#include <cstdint>
#include "mc2_log.h"

namespace mc2tiling {
// 根据 AIC/AIV 核数量按实际配比计算逻辑核数(blockDim)。
// 不同形态的 AIC:AIV 配比不同：A5 为 1:2、A6 为 1:1，
// 因此这里由 aivNum/aicNum 动态推导配比，取 aicNum 与 aivNum/ratio 的较小值作为基点。
inline uint64_t GetNumBlocks(const uint64_t aicNum, const uint64_t aivNum, const char* debugDesc)
{
    // 实际 AIC:AIV 配比（A5=1:2、A6=1:1）；aicNum 为 0 或 aivNum<aicNum 时按 1 兜底
    uint64_t ratio = (aicNum != 0) ? (aivNum / aicNum) : 1;
    ratio = (ratio == 0) ? 1 : ratio;
    uint64_t pairedAicNum = aivNum / ratio; // AIV 核能配对的 AIC 核数
    uint64_t numBlocks = std::min(aicNum, pairedAicNum);
    if (numBlocks != aicNum) { // AIV 核不足，无法配对所有 AIC 核
        OP_LOGI(debugDesc,
                "aicNum is %lu, aivNum is %lu, aic:aiv ratio is %lu. AIV cores can only pair %lu AIC cores, "
                "so the actual enabled AIC cores is %lu and AIV cores is %lu.",
                aicNum, aivNum, ratio, pairedAicNum, numBlocks, numBlocks * ratio);
    }
    if (numBlocks == 0U) {
        const char* opName = debugDesc == nullptr ? "Mc2Tiling" : debugDesc;
        const std::string reason = "GetNumBlocks: aic=" + std::to_string(aicNum) + ", aiv=" + std::to_string(aivNum) +
                                   ", ratio=" + std::to_string(ratio) +
                                   "; no logical core can be formed from the platform core counts";
        OP_LOGE(opName, "%s", reason.c_str());
        return 0;
    }
    return numBlocks;
}
} // namespace mc2tiling
#endif
