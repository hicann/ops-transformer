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
 * \file block_attention_residuals_infershape.cpp
 * \brief BlockAttentionResiduals infer shape implementation
 */
#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/shape.h"
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "err/ops_err.h"

using namespace gert;
namespace ops {

constexpr size_t PARTIAL_BLOCK_INDEX = 0;
constexpr size_t BLOCK_RES_INDEX = 1;
constexpr size_t PROJ_WEIGHT_INDEX = 2;
constexpr size_t NORM_WEIGHT_INDEX = 3;
constexpr size_t HIDDEN_STATES_INDEX = 0;
constexpr size_t INV_NORM_INDEX = 1;
constexpr size_t PROBS_INDEX = 2;

constexpr size_t PARTIAL_BLOCK_DIM = 2;
constexpr size_t BLOCK_RES_DIM = 3;
constexpr size_t WEIGHT_1D_DIM = 1;
constexpr size_t WEIGHT_2D_DIM = 2;
constexpr size_t ATTR_VALID_BLOCK_NUM_INDEX = 0;
constexpr size_t ATTR_NORM_EPS_INDEX = 1;
constexpr size_t ATTR_NEED_BACKWARD_INDEX = 2;
constexpr int64_t MAX_NUM_BLOCKS = 100;

static ge::graphStatus InferShapeBlockAttentionResiduals(InferShapeContext *context)
{
    if (context == nullptr) {
        OP_LOGE("BlockAttentionResiduals", "inference context is null");
        return ge::GRAPH_FAILED;
    }

    auto opName = context->GetNodeName();
    auto partialBlockShape = context->GetInputShape(PARTIAL_BLOCK_INDEX);
    auto blockShape = context->GetInputShape(BLOCK_RES_INDEX);
    auto projShape = context->GetInputShape(PROJ_WEIGHT_INDEX);
    auto normShape = context->GetInputShape(NORM_WEIGHT_INDEX);
    auto outShape = context->GetOutputShape(HIDDEN_STATES_INDEX);
    if (partialBlockShape == nullptr || blockShape == nullptr || projShape == nullptr || normShape == nullptr ||
        outShape == nullptr) {
        OP_LOGE(opName, "[InferShape] shape is null");
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF(partialBlockShape->GetDimNum() != PARTIAL_BLOCK_DIM,
                OP_LOGE(opName, "partial_block dim num should be %zu, got %zu", PARTIAL_BLOCK_DIM,
                        partialBlockShape->GetDimNum()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(blockShape->GetDimNum() != BLOCK_RES_DIM,
                OP_LOGE(opName, "block_res dim num should be %zu, got %zu", BLOCK_RES_DIM, blockShape->GetDimNum()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(projShape->GetDimNum() != WEIGHT_1D_DIM && projShape->GetDimNum() != WEIGHT_2D_DIM,
                OP_LOGE(opName, "proj_weight dim num should be 1 or 2, got %zu", projShape->GetDimNum()),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(normShape->GetDimNum() != WEIGHT_1D_DIM,
                OP_LOGE(opName, "norm_weight dim num should be %zu, got %zu", WEIGHT_1D_DIM, normShape->GetDimNum()),
                return ge::GRAPH_FAILED);

    const int64_t numTokens = partialBlockShape->GetDim(0);
    const int64_t hiddenSize = partialBlockShape->GetDim(1);
    const int64_t numBlocks = blockShape->GetDim(1);
    const int64_t blockCount = numBlocks + 1;

    OP_CHECK_IF(numTokens < 0 || hiddenSize <= 0 || numBlocks < 1 || numBlocks > MAX_NUM_BLOCKS,
                OP_LOGE(opName, "shape range requires T>=0, H>=1 and 1<=N<=%ld", MAX_NUM_BLOCKS),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(blockShape->GetDim(0) != numTokens || blockShape->GetDim(2) != hiddenSize,
                OP_LOGE(opName, "block_res shape must be [T,N,H]"), return ge::GRAPH_FAILED);
    const bool projShapeOk =
        (projShape->GetDimNum() == WEIGHT_1D_DIM && projShape->GetDim(0) == hiddenSize) ||
        (projShape->GetDimNum() == WEIGHT_2D_DIM && projShape->GetDim(0) == 1 && projShape->GetDim(1) == hiddenSize);
    OP_CHECK_IF(!projShapeOk, OP_LOGE(opName, "proj_weight shape must be [H] or [1,H]"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(normShape->GetDim(0) != hiddenSize, OP_LOGE(opName, "norm_weight shape must be [H]"),
                return ge::GRAPH_FAILED);

    outShape->SetDimNum(PARTIAL_BLOCK_DIM);
    outShape->SetDim(0, numTokens);
    outShape->SetDim(1, hiddenSize);

    bool needBackward = false;
    auto attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const bool *needBackwardPtr = attrs->GetAttrPointer<bool>(ATTR_NEED_BACKWARD_INDEX);
        if (needBackwardPtr != nullptr) {
            needBackward = *needBackwardPtr;
        }
    }

    if (needBackward) {
        auto invNormShape = context->GetOutputShape(INV_NORM_INDEX);
        auto probsShape = context->GetOutputShape(PROBS_INDEX);
        if (invNormShape != nullptr) {
            invNormShape->SetDimNum(PARTIAL_BLOCK_DIM);
            invNormShape->SetDim(0, numTokens);
            invNormShape->SetDim(1, blockCount);
        }
        if (probsShape != nullptr) {
            probsShape->SetDimNum(PARTIAL_BLOCK_DIM);
            probsShape->SetDim(0, numTokens);
            probsShape->SetDim(1, blockCount);
        }
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeBlockAttentionResiduals(gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        OP_LOGE("BlockAttentionResiduals", "inference context is null");
        return ge::GRAPH_FAILED;
    }
    auto partialBlockDtype = context->GetInputDataType(PARTIAL_BLOCK_INDEX);
    context->SetOutputDataType(HIDDEN_STATES_INDEX, partialBlockDtype);

    bool needBackward = false;
    auto attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const bool *needBackwardPtr = attrs->GetAttrPointer<bool>(ATTR_NEED_BACKWARD_INDEX);
        if (needBackwardPtr != nullptr) {
            needBackward = *needBackwardPtr;
        }
    }
    if (needBackward) {
        context->SetOutputDataType(INV_NORM_INDEX, ge::DT_FLOAT);
        context->SetOutputDataType(PROBS_INDEX, ge::DT_FLOAT);
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(BlockAttentionResiduals)
    .InferShape(InferShapeBlockAttentionResiduals)
    .InferDataType(InferDataTypeBlockAttentionResiduals);
} // namespace ops
