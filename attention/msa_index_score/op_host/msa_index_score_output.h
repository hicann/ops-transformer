/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MSA_INDEX_SCORE_OUTPUT_H
#define MSA_INDEX_SCORE_OUTPUT_H

#include <limits>
#include "exe_graph/runtime/extended_kernel_context.h"
#include "exe_graph/runtime/storage_shape.h"

namespace MsaIndexScoreOutput {

constexpr size_t SCORE = 0;
constexpr size_t INDICES = 1;
constexpr size_t VALUES = 2;
constexpr size_t ABSENT = std::numeric_limits<size_t>::max();

// Runtime output indices can shift when an optional IR output is omitted.
inline size_t RuntimeIndex(const gert::ExtendedKernelContext* context, size_t irIndex)
{
    const auto* instance = context->GetIrOutputInstanceInfo(irIndex);
    if (instance != nullptr) {
        return instance->GetInstanceNum() == 0U ? ABSENT : instance->GetInstanceStart();
    }
    const auto* node = context->GetComputeNodeInfo();
    return node != nullptr && node->GetIrOutputsNum() == 0U && irIndex < node->GetOutputsNum() ? irIndex : ABSENT;
}

inline bool IsPresent(const gert::StorageShape* shape)
{
    if (shape == nullptr) {
        return false;
    }
    const auto& storageShape = shape->GetStorageShape();
    return storageShape.GetDimNum() != 0U && !(storageShape.GetDimNum() == 1U && storageShape.GetDim(0) == 0);
}

inline void SetShape(gert::Shape* shape, bool enabled, int64_t heads, int64_t totalQ, int64_t width)
{
    if (shape != nullptr) {
        *shape = enabled ? gert::Shape{heads, totalQ, width} : gert::Shape{0};
    }
}

} // namespace MsaIndexScoreOutput

#endif // MSA_INDEX_SCORE_OUTPUT_H
