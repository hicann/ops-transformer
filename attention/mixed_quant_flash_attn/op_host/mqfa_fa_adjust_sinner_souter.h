/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MQFA_FA_ADJUST_SINNER_SOUTER_H
#define MQFA_FA_ADJUST_SINNER_SOUTER_H

#include <cstdint>

namespace optiling {
namespace mixed_quant_flash_attn {
namespace fa_tiling_util {

constexpr int64_t MAX_SEQ_LEN_DEFAULT = 2147483647;
// layout 枚举值，与 FaLayout 一致，供外部算子使用
constexpr uint32_t LAYOUT_BSH = 0;
constexpr uint32_t LAYOUT_BSND = 0;
constexpr uint32_t LAYOUT_BNSD = 1;
constexpr uint32_t LAYOUT_TND = 2;

// tiling 切块常量
constexpr uint32_t SOUTER_32 = 32;
constexpr uint32_t SOUTER_48 = 48;
constexpr uint32_t SOUTER_64 = 64;
constexpr uint32_t SINNER_128 = 128;
constexpr uint32_t SINNER_256 = 256;
constexpr uint32_t SINNER_512 = 512;
constexpr uint32_t DSIZE_128 = 128;
constexpr uint32_t DSIZE_256 = 256;

/**
 * @brief 根据算子参数决定 sOuter / sInner 切块大小，纯函数，不依赖任何类。
 *
 * @param vHeadDim   V 的 head dim
 * @param s1Size     Q 的序列长度
 * @param s2Size     KV 的序列长度
 * @param maskMode   mask 模式（0/2/4 等）
 * @param winLeft    左侧窗口
 * @param winRight   右侧窗口
 * @param qLayout    Q 的 layout（使用 LAYOUT_BSH / LAYOUT_BSND / LAYOUT_TND 等）
 * @param sOuterFactor [out] sOuter 切块大小
 * @param sInnerFactor [out] sInner 切块大小
 */
inline void AdjustSinnerAndSouter(uint32_t vHeadDim, uint32_t gSize, int64_t maxSeqQ, int64_t maxSeqKv,
                                  int32_t maskMode, int64_t winLeft, int64_t winRight, uint32_t qLayout,
                                  uint32_t& sOuterFactor, uint32_t& sInnerFactor)
{
    if (maxSeqQ != -1) {
        if (maxSeqQ * gSize <= 32) {
            sOuterFactor = SOUTER_32;
            sInnerFactor = SINNER_512;
        } else if (maxSeqQ * gSize <= 48) {
            sOuterFactor = SOUTER_48;
            sInnerFactor = SINNER_512;
        } else if (maxSeqQ * gSize <= 64) {
            sOuterFactor = SOUTER_64;
            sInnerFactor = SINNER_512;
        } else {
            sOuterFactor = SOUTER_48;
            sInnerFactor = SINNER_512;
        }
    } else {
        sOuterFactor = SOUTER_48;
        sInnerFactor = SINNER_512;
    }
}

} // namespace fa_tiling_util
} // namespace mixed_quant_flash_attn
} // namespace optiling

#endif // MQFA_FA_ADJUST_SINNER_SOUTER_H
