/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATTENTION_UPDATE_LAYOUT_H_
#define ATTENTION_UPDATE_LAYOUT_H_

#include <limits>
#include "exe_graph/runtime/tiling_context.h"
#include "log/log.h"

namespace optiling {
// Missing metadata is accepted only for an ordinary dense tensor. A real view
// must carry its stride; losing it must not silently select the dense kernel.
inline ge::graphStatus GetAttentionUpdateRowStride(gert::TilingContext* context, uint64_t sp, uint64_t rows, uint64_t d,
                                                   uint64_t elementBytes, uint64_t& rowStride)
{
    rowStride = d;
    for (uint64_t p = 0; p < sp; ++p) {
        auto lseShape = context->GetDynamicInputShape(0, p);
        if (lseShape == nullptr || lseShape->GetOriginShape().GetDimNum() != 1 ||
            lseShape->GetOriginShape().GetDim(0) != static_cast<int64_t>(rows)) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                context->GetNodeName(), "lse",
                lseShape == nullptr ? "null" : Ops::Base::ToString(lseShape->GetOriginShape()),
                "Each shard must have shape [T], T=" + std::to_string(rows));
            return ge::GRAPH_FAILED;
        }
        auto lseStride = context->GetDynamicInputStride(0, p);
        if ((lseStride == nullptr || lseStride->GetDimNum() == 0) &&
            lseShape->GetStorageShape() != lseShape->GetOriginShape()) {
            OP_LOGE_FOR_INVALID_STRIDE(context->GetNodeName(), "lse", "missing", "[1] for a view");
            return ge::GRAPH_FAILED;
        }
        if (lseStride != nullptr && lseStride->GetDimNum() != 0 &&
            (lseStride->GetDimNum() != 1 || (rows > 1 && lseStride->GetStride(0) != 1))) {
            OP_LOGE_FOR_INVALID_STRIDE(context->GetNodeName(), "lse",
                                       "rank=" + std::to_string(lseStride->GetDimNum()) +
                                           ", stride[0]=" + std::to_string(lseStride->GetStride(0)),
                                       "rank 1, stride [1] when T > 1");
            return ge::GRAPH_FAILED;
        }
        auto storage = context->GetDynamicInputShape(1, p);
        if (storage == nullptr || storage->GetOriginShape().GetDimNum() != 2 ||
            storage->GetOriginShape().GetDim(0) != static_cast<int64_t>(rows) ||
            storage->GetOriginShape().GetDim(1) != static_cast<int64_t>(d)) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                context->GetNodeName(), "localOut",
                storage == nullptr ? "null" : Ops::Base::ToString(storage->GetOriginShape()),
                "Each shard must have shape [T,D], T=" + std::to_string(rows) + ", D=" + std::to_string(d));
            return ge::GRAPH_FAILED;
        }
        const auto* stride = context->GetDynamicInputStride(1, p);
        uint64_t currentStride = d;
        if (stride == nullptr || stride->GetDimNum() == 0) {
            if (storage->GetStorageShape() != storage->GetOriginShape()) {
                OP_LOGE_FOR_INVALID_STRIDE(context->GetNodeName(), "localOut", "missing", "[rowStride,1] for a view");
                return ge::GRAPH_FAILED;
            }
        } else {
            if (stride->GetDimNum() != 2 || (d > 1 && stride->GetStride(1) != 1) ||
                (rows > 1 && stride->GetStride(0) < static_cast<int64_t>(d))) {
                OP_LOGE_FOR_INVALID_STRIDE(
                    context->GetNodeName(), "localOut",
                    "rank=" + std::to_string(stride->GetDimNum()) + ", row=" + std::to_string(stride->GetStride(0)) +
                        ", column=" + (stride->GetDimNum() > 1 ? std::to_string(stride->GetStride(1)) : "missing"),
                    "rank 2, column stride 1 when D > 1, row stride >= D when T > 1");
                return ge::GRAPH_FAILED;
            }
            if (rows > 1) {
                currentStride = static_cast<uint64_t>(stride->GetStride(0));
            }
        }
        if (p != 0 && currentStride != rowStride) {
            OP_LOGE_FOR_INVALID_STRIDE(context->GetNodeName(), "localOut", std::to_string(currentStride),
                                       "All shards must have row stride " + std::to_string(rowStride));
            return ge::GRAPH_FAILED;
        }
        rowStride = currentStride;
    }
    const uint64_t maxElements = static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) / elementBytes;
    if (d > maxElements || (rows > 1 && rowStride > (maxElements - d) / (rows - 1))) {
        OP_LOGE_FOR_INVALID_VALUES_WITH_REASON(
            context->GetNodeName(), "localOut T,D,rowStride,elementBytes",
            std::to_string(rows) + "," + std::to_string(d) + "," + std::to_string(rowStride) + "," +
                std::to_string(elementBytes),
            "The byte address range must fit in int64; reduce the shape or row stride");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
#endif
