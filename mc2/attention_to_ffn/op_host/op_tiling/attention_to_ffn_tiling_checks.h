/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATTENTION_TO_FFN_TILING_CHECKS_H
#define ATTENTION_TO_FFN_TILING_CHECKS_H

#include <string>
#include "register/tilingdata_base.h"
#include "mc2_log.h"

namespace MC2Tiling {
constexpr uint32_t UNQUANT_MODE = 0;
constexpr uint32_t DYNAMIC_QUANT_MODE = 2;
constexpr uint32_t NO_SCALES = 0;
constexpr uint32_t DYNAMIC_SCALES = 2;
constexpr int64_t MOE_EXPERT_MAX_NUM = 1024;
constexpr int64_t MIN_WORLD_SIZE = 2;
constexpr uint32_t USER_QUANT_MODE_MX_E5M2 = 3U;
constexpr uint32_t USER_QUANT_MODE_MX_E4M3 = 4U;
constexpr uint32_t USER_QUANT_MODE_MX_E2M1 = 5U;
constexpr uint32_t USER_QUANT_MODE_MX_CLIP_E5M2 = 6U;
constexpr uint32_t USER_QUANT_MODE_MX_CLIP_E4M3 = 7U;

inline bool IsMxQuantMode(int64_t quantMode)
{
    return quantMode == USER_QUANT_MODE_MX_E5M2 || quantMode == USER_QUANT_MODE_MX_E4M3 ||
           quantMode == USER_QUANT_MODE_MX_E2M1 || quantMode == USER_QUANT_MODE_MX_CLIP_E5M2 ||
           quantMode == USER_QUANT_MODE_MX_CLIP_E4M3;
}

inline bool CheckAttrValues(const char* nodeName, const int64_t moeExpertNum, const int64_t worldSize,
                            const int64_t quantMode, const int64_t syncFlag, const int64_t ffnStartRankId,
                            const bool allowMxQuantMode, const int64_t maxWorldSize)
{
    OP_TILING_CHECK(
        (worldSize < MIN_WORLD_SIZE) || (static_cast<int64_t>(worldSize) > maxWorldSize),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            nodeName, "world_size", std::to_string(worldSize).c_str(),
            (std::string("[") + std::to_string(MIN_WORLD_SIZE) + ", " + std::to_string(maxWorldSize) + "]").c_str()),
        return false);

    OP_TILING_CHECK(
        (moeExpertNum <= 0) || (moeExpertNum > MOE_EXPERT_MAX_NUM),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "moe_expert_num", std::to_string(moeExpertNum).c_str(),
                                              (std::string("(0, ") + std::to_string(MOE_EXPERT_MAX_NUM) + "]").c_str()),
        return false);
    if (allowMxQuantMode) {
        OP_TILING_CHECK(quantMode != UNQUANT_MODE && quantMode != DYNAMIC_QUANT_MODE &&
                            !IsMxQuantMode(static_cast<uint32_t>(quantMode)),
                        OP_LOGE_WITH_INVALID_ATTR("AttentionToFFN Tiling Debug", "quant_mode",
                                                  std::to_string(quantMode).c_str(), "0, 2, 3, 4, 5, 6 or 7"),
                        return false);
    } else {
        OP_TILING_CHECK(
            (quantMode < static_cast<int64_t>(NO_SCALES)) || (quantMode > static_cast<int64_t>(DYNAMIC_SCALES)),
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "quant_mode", std::to_string(quantMode).c_str(),
                                                  (std::string("[0, ") + std::to_string(DYNAMIC_SCALES) + "]").c_str()),
            return false);
    }
    OP_TILING_CHECK(
        (syncFlag != 0) && (syncFlag != 1),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "sync_flag", std::to_string(syncFlag).c_str(), "0 or 1"),
        return false);
    OP_TILING_CHECK(
        (ffnStartRankId >= worldSize) || (ffnStartRankId < 0),
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(nodeName, "ffn_start_rank_id", std::to_string(ffnStartRankId).c_str(),
                                              (std::string("[0, ") + std::to_string(worldSize) + ")").c_str()),
        return false);

    return true;
}
} // namespace MC2Tiling

#endif // ATTENTION_TO_FFN_TILING_CHECKS_H
