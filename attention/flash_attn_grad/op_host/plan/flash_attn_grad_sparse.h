/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef FLASH_ATTN_GRAD_SPARSE_H_
#define FLASH_ATTN_GRAD_SPARSE_H_

#include <algorithm>
#include <cstdint>
#include <limits>

#include "../info/flash_attn_grad_tiling_info.h"
#include "flash_attn_grad_block_outer.h"

namespace optiling {

// SparseType: 0=DENSE, 1=CASUAL, 2=BAND.
constexpr int64_t FAG_SPARSE_DENSE = 0;
constexpr int64_t FAG_SPARSE_CASUAL = 1;
constexpr int64_t FAG_SPARSE_BAND = 2;
constexpr int64_t FAG_CUBE_BASE = 128;

inline void FagCorrectSparseTokens(int64_t maskMode, int64_t winLeft, int64_t winRight, int64_t s1, int64_t s2,
                                   int64_t &s1Token, int64_t &s2Token)
{
    // ProcessTokensInfo (non-TND): causal tokens then right-down pad correction.
    if (maskMode == MASK_MODE_CAUSAL) {
        s1Token = static_cast<int64_t>(std::numeric_limits<int32_t>::max());
        s2Token = 0;
    } else if (maskMode == MASK_MODE_WINDOW) {
        s1Token = winLeft;
        s2Token = winRight;
    } else {
        s1Token = static_cast<int64_t>(std::numeric_limits<int32_t>::max());
        s2Token = static_cast<int64_t>(std::numeric_limits<int32_t>::max());
        return;
    }
    s1Token = s1Token + s1 - s2;
    s2Token = s2Token - s1 + s2;
}

inline int64_t FagDecideSparseType(int64_t maskMode, int64_t s1, int64_t s2, int64_t s1Token, int64_t s2Token)
{
    // GetSparseType non-TND. Mode 3/4 only; no ALL_MASK / NO_MASK / TND.
    if (maskMode != MASK_MODE_CAUSAL && maskMode != MASK_MODE_WINDOW) {
        return FAG_SPARSE_DENSE;
    }
    const bool casualCondition =
        (maskMode == MASK_MODE_CAUSAL && s1 <= s2) || (maskMode == MASK_MODE_WINDOW && s1Token >= s1 && s2Token == 0);
    if (casualCondition) {
        return FAG_SPARSE_CASUAL;
    }
    return FAG_SPARSE_BAND;
}

// 空 s2 列或从未被覆盖的 s1 行。
// Host ORs the two; kernel pre zeros all three out tensors.
inline bool FagBn2HasInvalidOuter(int64_t s1, int64_t s2, int64_t s1Token, int64_t s2Token)
{
    int64_t s1Outer = 0;
    int64_t s2Outer = 0;
    FagS1S2Outer(s1, s2, s1Outer, s2Outer);
    if (s1Outer <= 0 || s2Outer <= 0) {
        return false;
    }
    constexpr int64_t kMaxOuter = 16;
    if (s1Outer > kMaxOuter || s2Outer > kMaxOuter) {
        return true;
    }
    int64_t s1CvInner = FagCvInner(s1, FAG_S1_INNER, FAG_S1CV_RATIO_DEFAULT);
    int64_t cvS2Inner = FagCvInner(s2, FAG_S2_INNER, FAG_S2CV_RATIO_DEFAULT);
    int64_t s2CvTail = s2 - (s2Outer - 1) * cvS2Inner;
    if (s2CvTail <= 0) {
        s2CvTail = cvS2Inner;
    }
    bool s1Seen[kMaxOuter] = {};
    bool invalidCol = false;
    for (int64_t i = 0; i < s2Outer; ++i) {
        int64_t leftIntersectionPoint = std::max(int64_t(0), cvS2Inner * i - s2Token);
        int64_t beginIdx = 0;
        if (leftIntersectionPoint > s1) {
            beginIdx = (s1 + s1CvInner - 1) / s1CvInner;
        } else {
            beginIdx = leftIntersectionPoint / s1CvInner;
        }
        int64_t cvBlockTail = (i == s2Outer - 1) ? s2CvTail : cvS2Inner;
        int64_t endIdx =
            (std::min(std::max(int64_t(0), cvS2Inner * i + cvBlockTail + s1Token), s1) + s1CvInner - 1) / s1CvInner;
        if (beginIdx >= endIdx) {
            invalidCol = true;
        }
        for (int64_t j = 0; j < s1Outer; ++j) {
            if (j >= beginIdx && j < endIdx) {
                s1Seen[j] = true;
            }
        }
    }
    for (int64_t j = 0; j < s1Outer; ++j) {
        if (!s1Seen[j]) {
            return true;
        }
    }
    return invalidCol;
}

inline int64_t FagGetTotalPerBatchNum(int64_t s1, int64_t s2, int64_t s1Outer, int64_t s2Outer, int64_t sparseType,
                                      int64_t maskMode, int64_t s1Token, int64_t s2Token)
{
    if (sparseType == FAG_SPARSE_DENSE) {
        return s1Outer * s2Outer;
    }
    if (sparseType == FAG_SPARSE_CASUAL) {
        if (s1 < s2) {
            if (maskMode == MASK_MODE_CAUSAL) {
                return ((((s2Outer << 1) - s1Outer + 1) * s1Outer) >> 1) + (s1Outer - 1);
            }
            return (((s1Outer << 1) - s1Outer + 1) * s1Outer) >> 1;
        }
        return (((s1Outer << 1) - s2Outer + 1) * s2Outer) >> 1;
    }
    int64_t p = FagCeilDiv(s1Token, FAG_CUBE_BASE);
    int64_t q = FagCeilDiv(s2Token, FAG_CUBE_BASE);
    int64_t total = 0;
    for (int64_t s2oIdx = 0; s2oIdx < s2Outer; ++s2oIdx) {
        int64_t xMin = (s2oIdx - q) > 0 ? (s2oIdx - q) : 0;
        int64_t xMax = (s1Outer - 1) > (s2oIdx + p) ? (s2oIdx + p) : (s1Outer - 1);
        int64_t length = (xMax >= xMin) ? (xMax - xMin + 1) : 0;
        if (length > 0) {
            total += length;
        }
    }
    return total;
}

inline void FagFillSparseInfo(FagParsedInfo &info)
{
    FagCorrectSparseTokens(info.maskMode, info.winLeft, info.winRight, info.s1, info.s2, info.s1Token, info.s2Token);
    info.sparseType = FagDecideSparseType(info.maskMode, info.s1, info.s2, info.s1Token, info.s2Token);
    int64_t s1Outer = 0;
    int64_t s2Outer = 0;
    FagS1S2Outer(info.s1, info.s2, s1Outer, s2Outer);
    info.totalPerBatchNum = FagGetTotalPerBatchNum(info.s1, info.s2, s1Outer, s2Outer, info.sparseType, info.maskMode,
                                                   info.s1Token, info.s2Token);
    if (info.totalPerBatchNum <= 0) {
        info.totalPerBatchNum = s1Outer * s2Outer;
    }
}

} // namespace optiling

#endif // FLASH_ATTN_GRAD_SPARSE_H_
