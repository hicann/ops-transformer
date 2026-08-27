/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QUANT_FLASH_ATTN_QFA_ADJUST_SINNER_SOUTER_H
#define QUANT_FLASH_ATTN_QFA_ADJUST_SINNER_SOUTER_H

#include <cstdint>

namespace optiling {
namespace quant_flash_attn {
namespace qfa_tiling_util {

constexpr int64_t MAX_SEQ_LEN_DEFAULT = 2147483647;

constexpr uint32_t LAYOUT_BSH = 0;
constexpr uint32_t LAYOUT_BSND = 0;
constexpr uint32_t LAYOUT_BNSD = 1;
constexpr uint32_t LAYOUT_TND = 2;

constexpr uint32_t SOUTER_64 = 64;
constexpr uint32_t SINNER_256 = 256;
constexpr uint32_t SINNER_512 = 512;
constexpr uint32_t DSIZE_256 = 256;

/**
 * @brief 根据算子参数决定 sOuter / sInner 切块大小，纯函数，不依赖任何类。
 *
 * @param vHeadDim   V 的 head dim
 * @param maxSeqQ    Q 的 max sequence length，-1 表示未知（按极大值处理）
 * @param maxSeqKv   KV 的 max sequence length，-1 表示未知（按极大值处理）
 * @param maskMode   mask 模式（0/2/4 等）
 * @param winLeft    左侧窗口，调用方直接传入接口值，-1 表示无限制，函数内部会转为正无穷
 * @param winRight   右侧窗口，调用方直接传入接口值，-1 表示无限制，函数内部会转为正无穷
 * @param qLayout    Q 的 layout（使用 LAYOUT_BSH / LAYOUT_BSND / LAYOUT_TND 等）
 * @param quantMode  量化模式（0=HIF8, 1=MXFP8 softmax FP32）
 * @param sOuterFactor [out] sOuter 切块大小
 * @param sInnerFactor [out] sInner 切块大小
 */
inline void AdjustSinnerAndSouter(uint32_t vHeadDim, int64_t maxSeqQ, int64_t maxSeqKv, int32_t maskMode,
                                  int64_t winLeft, int64_t winRight, uint32_t qLayout, uint32_t quantMode,
                                  uint32_t &sOuterFactor, uint32_t &sInnerFactor)
{
    if (maxSeqQ == -1) {
        maxSeqQ = MAX_SEQ_LEN_DEFAULT;
    }
    if (maxSeqKv == -1) {
        maxSeqKv = MAX_SEQ_LEN_DEFAULT;
    }
    if (winLeft == -1) {
        winLeft = MAX_SEQ_LEN_DEFAULT;
    }
    if (winRight == -1) {
        winRight = MAX_SEQ_LEN_DEFAULT;
    }
    if (vHeadDim == DSIZE_256) {
        sOuterFactor = SOUTER_64;
        sInnerFactor = SINNER_256;
    } else if (quantMode == 0 || quantMode == 6) { // QFA_HIF8_FP32 or GQA_FP8
        sOuterFactor = SOUTER_64;
        sInnerFactor = SINNER_256;
    } else {
        sOuterFactor = SOUTER_64;
        sInnerFactor = SINNER_512;
    }
}

} // namespace qfa_tiling_util
} // namespace quant_flash_attn
} // namespace optiling

#endif // QUANT_FLASH_ATTN_QFA_ADJUST_SINNER_SOUTER_H
