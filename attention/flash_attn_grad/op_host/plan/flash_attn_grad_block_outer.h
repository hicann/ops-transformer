/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef FLASH_ATTN_GRAD_BLOCK_OUTER_H_
#define FLASH_ATTN_GRAD_BLOCK_OUTER_H_

#include <algorithm>
#include <cstdint>

namespace optiling {

// Must match flash_attn_grad_metadata_split.h InitFagBaseDims.
constexpr int64_t FAG_S1_INNER = 64;
constexpr int64_t FAG_S2_INNER = 128;
constexpr int64_t FAG_S1CV_RATIO_DEFAULT = 2;
constexpr int64_t FAG_S2CV_RATIO_DEFAULT = 1;
constexpr uint32_t FAG_AICV_RATIO_DEFAULT = 2;

inline int64_t FagCeilDiv(int64_t a, int64_t b)
{
    return (b == 0) ? 0 : (a + b - 1) / b;
}

inline int64_t FagCvInner(int64_t seq, int64_t inner, int64_t ratio)
{
    if (seq == 0) {
        return 1;
    }
    return (inner * ratio > seq) ? seq : inner * ratio;
}

inline void FagS1S2Outer(int64_t s1, int64_t s2, int64_t &s1Outer, int64_t &s2Outer)
{
    int64_t s1CvInner = FagCvInner(s1, FAG_S1_INNER, FAG_S1CV_RATIO_DEFAULT);
    int64_t s2CvInner = FagCvInner(s2, FAG_S2_INNER, FAG_S2CV_RATIO_DEFAULT);
    s1Outer = (s1 == 0) ? 0 : FagCeilDiv(s1, s1CvInner);
    s2Outer = (s2 == 0) ? 0 : FagCeilDiv(s2, s2CvInner);
}

// ceil(fused / ceil(fused / aicNum)) == actual used cube cores.
// Same identity as DoFagDenseSplit / DoFagBn2DenseSplit.
inline int64_t FagDenseBlockOuter(int64_t fusedOuter, int64_t aicNum)
{
    if (fusedOuter <= 0 || aicNum <= 0) {
        return 0;
    }
    int64_t blockFactor = FagCeilDiv(fusedOuter, aicNum);
    return FagCeilDiv(fusedOuter, blockFactor);
}

// BN2: fusedOuter = b*n2*g (blockOuter counted on BN, not S tiles).
// BN2GS1S2 dense: fusedOuter = b*n2*g*s1Outer*s2Outer.
inline int64_t FagUsedCubeCores(bool isBn2, int64_t b, int64_t n2, int64_t g, int64_t s1Outer, int64_t s2Outer,
                                int64_t aicNum)
{
    int64_t fusedOuter = isBn2 ? (b * n2 * g) : (b * n2 * g * s1Outer * s2Outer);
    int64_t blockOuter = FagDenseBlockOuter(fusedOuter, aicNum);
    if (blockOuter <= 0) {
        return 1;
    }
    return std::min(blockOuter, aicNum);
}

} // namespace optiling

#endif // FLASH_ATTN_GRAD_BLOCK_OUTER_H_
