/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file engram_hash_infershape.cpp
 * \brief EngramHash shape/dtype 推导
 */
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "log/log.h"
#include "util/shape_util.h"

using namespace ge;
namespace ops {

graphStatus InferShape4EngramHash(gert::InferShapeContext* context)
{
    // 输出 hash_indices shape 为 [L, T, C]:
    //   L = vocab_sizes.dim(0), T = ngram_token_ids.dim(0), C = (N - 1) * tables。
    const gert::Shape* idsShape = context->GetInputShape(0);
    const gert::Shape* multShape = context->GetInputShape(1);
    const gert::Shape* vocabShape = context->GetInputShape(2);
    const gert::Shape* offsShape = context->GetInputShape(3);
    gert::Shape* outShape = context->GetOutputShape(0);
    if (idsShape == nullptr || multShape == nullptr || vocabShape == nullptr || offsShape == nullptr ||
        outShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    // 动态 shape: 任一输入未知秩 (-2) 时输出整体设为未知秩, 跳过后续校验
    // (同 gather_pa_kv_cache / kv_rms_norm_rope_cache 的 IsUnknownRank 处理惯例)。
    for (uint32_t i = 0; i < context->GetComputeNodeInputNum(); i++) {
        const gert::Shape* shape = context->GetInputShape(i);
        if (shape != nullptr && Ops::Base::IsUnknownRank(*shape)) {
            Ops::Base::SetUnknownRank(*outShape);
            OP_LOGD(context, "input[%u] shape is UnknownRank, set output hash_indices to UnknownRank.", i);
            return ge::GRAPH_SUCCESS;
        }
    }
    // 未知维度 (-1): 已知部分照常传播到输出 (T/L 直接透传, C 仅在 N/tables 已知时推导),
    // 跳过结构校验, 交由执行期 tiling 校验。
    if (idsShape->GetDimNum() == 2 && vocabShape->GetDimNum() == 3 &&
        (Ops::Base::IsUnknownShape(*idsShape) || Ops::Base::IsUnknownShape(*vocabShape) ||
         Ops::Base::IsUnknownShape(*multShape) || Ops::Base::IsUnknownShape(*offsShape))) {
        const int64_t dynT = idsShape->GetDim(0);
        const int64_t dynN = idsShape->GetDim(1);
        const int64_t dynL = vocabShape->GetDim(0);
        const int64_t dynTables = vocabShape->GetDim(2);
        const int64_t dynC = (dynN >= 2 && dynTables > 0) ? (dynN - 1) * dynTables : -1;
        *outShape = {dynL > 0 ? dynL : -1, dynT, dynC};
        OP_LOGD(context, "input has UnknownShape (-1) dims, propagate known dims to output.");
        return ge::GRAPH_SUCCESS;
    }
    // shape 链校验 (空输出时 tiling 可能被跳过, 故在 InferShape 层也做完整校验,
    // 保证非法输入在 GetWorkspaceSize 阶段即被拒绝)。
    if (idsShape->GetDimNum() != 2 || vocabShape->GetDimNum() != 3) {
        return ge::GRAPH_FAILED;
    }
    const int64_t numTokens = idsShape->GetDim(0);
    const int64_t maxNgramSize = idsShape->GetDim(1);
    const int64_t numLayers = vocabShape->GetDim(0);
    const int64_t numTables = vocabShape->GetDim(2);
    if (maxNgramSize < 2 || numLayers <= 0 || numTables <= 0 || vocabShape->GetDim(1) != maxNgramSize - 1) {
        return ge::GRAPH_FAILED;
    }
    const int64_t numOutCols = (maxNgramSize - 1) * numTables;
    if (multShape->GetDimNum() != 2 || multShape->GetDim(0) != numLayers || multShape->GetDim(1) != maxNgramSize) {
        return ge::GRAPH_FAILED;
    }
    if (offsShape->GetDimNum() != 2 || offsShape->GetDim(0) != numLayers || offsShape->GetDim(1) != numOutCols) {
        return ge::GRAPH_FAILED;
    }
    *outShape = {numLayers, numTokens, numOutCols};
    return ge::GRAPH_SUCCESS;
}

graphStatus InferDtype4EngramHash(gert::InferDataTypeContext* context)
{
    // 输出固定为 INT32 (hash 余数 + 偏移均在 int32 范围内)。
    context->SetOutputDataType(0, ge::DT_INT32);
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(EngramHash).InferShape(InferShape4EngramHash).InferDataType(InferDtype4EngramHash);
} // namespace ops
