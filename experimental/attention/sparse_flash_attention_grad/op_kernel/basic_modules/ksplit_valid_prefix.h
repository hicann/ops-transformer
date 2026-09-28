/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <cstdint>

namespace SFAG_BASIC {

// For blockSize=1 and unique, legal causal indices, selecting visibleKeys
// entries covers every visible KV. Padding after that prefix is allowed.
__aicore__ inline bool KsplitIsFullCausalSelection(int64_t visibleKeys, int32_t rowActual, int32_t capacity)
{
    return visibleKeys > 0 && visibleKeys <= capacity && rowActual == visibleKeys;
}

// IndexTensor supplies GetValue(offset), as AscendC::GlobalTensor<int32_t> does.
// The valid/padding predicate must be monotone; valid indices need not be consecutive.
template <typename IndexTensor>
__aicore__ inline int32_t KsplitValidPrefixCount(IndexTensor &indices, int64_t base, int32_t upper)
{
    if (upper <= 0 || indices.GetValue(base) == -1) {
        return 0;
    }
    if (indices.GetValue(base + upper - 1) != -1) {
        return upper;
    }
    int32_t first = 1;
    int32_t last = upper - 1;
    while (first < last) {
        const int32_t mid = first + (last - first) / 2;
        if (indices.GetValue(base + mid) == -1) {
            last = mid;
        } else {
            first = mid + 1;
        }
    }
    return first;
}

} // namespace SFAG_BASIC
