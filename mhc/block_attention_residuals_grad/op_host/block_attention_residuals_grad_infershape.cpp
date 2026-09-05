/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {
namespace {
// Input tensor positions (matching the op def registration order).
constexpr size_t INPUT_PARTIAL_BLOCK = 0;
constexpr size_t INPUT_BLOCK_RES = 1;
constexpr size_t INPUT_PROJ_WEIGHT = 2;
constexpr size_t INPUT_NORM_WEIGHT = 3;
constexpr size_t INPUT_GRAD_HIDDEN_STATES = 4;
constexpr size_t INPUT_INV_NORM = 5;
constexpr size_t INPUT_PROBS = 6;

// Output tensor positions.
constexpr size_t OUTPUT_GRAD_PARTIAL_BLOCK = 0;
constexpr size_t OUTPUT_GRAD_BLOCK_RES = 1;
constexpr size_t OUTPUT_GRAD_PROJ_WEIGHT = 2;
constexpr size_t OUTPUT_GRAD_NORM_WEIGHT = 3;

// Fixed tensor ranks of the op.
constexpr size_t RANK_1D = 1;
constexpr size_t RANK_2D = 2;
constexpr size_t RANK_3D = 3;

// Shape axis positions of the fixed layouts:
//   [B, H]      partial_block / grad_hidden_states
//   [B, N, H]   block_res
//   [1, H]      proj_weight
//   [H]         norm_weight
//   [B, N + 1]  inv_norm / probs
constexpr size_t DIM_BATCH = 0;
constexpr size_t DIM_HIDDEN = 1; // [B, H] / [1, H] 的 H 轴
constexpr size_t DIM_BLOCK = 1;  // [B, N, H] 的 N 轴；[B, N + 1] 的 N + 1 轴
constexpr size_t DIM_BLOCK_RES_HIDDEN = 2;
constexpr size_t DIM_WEIGHT_ROW = 0;  // proj_weight [1, H] 的固定 1 轴
constexpr size_t DIM_NORM_WEIGHT = 0; // norm_weight [H] 的唯一轴
} // namespace

static bool CheckShapeBlockAttentionResidualsGrad(gert::InferShapeContext *context, int64_t &B, int64_t &N, int64_t &H)
{
    const gert::Shape *psShape = context->GetInputShape(INPUT_PARTIAL_BLOCK);
    const gert::Shape *brShape = context->GetInputShape(INPUT_BLOCK_RES);
    const gert::Shape *pwShape = context->GetInputShape(INPUT_PROJ_WEIGHT);
    const gert::Shape *nwShape = context->GetInputShape(INPUT_NORM_WEIGHT);
    const gert::Shape *gradHiddenStateShape = context->GetInputShape(INPUT_GRAD_HIDDEN_STATES);
    const gert::Shape *irShape = context->GetInputShape(INPUT_INV_NORM);
    const gert::Shape *pbShape = context->GetInputShape(INPUT_PROBS);

    if (psShape->GetDimNum() != RANK_2D || brShape->GetDimNum() != RANK_3D || pwShape->GetDimNum() != RANK_2D ||
        nwShape->GetDimNum() != RANK_1D || gradHiddenStateShape->GetDimNum() != RANK_2D ||
        irShape->GetDimNum() != RANK_2D || pbShape->GetDimNum() != RANK_2D) {
        return false;
    }

    B = psShape->GetDim(DIM_BATCH);
    H = psShape->GetDim(DIM_HIDDEN);
    N = brShape->GetDim(DIM_BLOCK);

    if (brShape->GetDim(DIM_BATCH) != B || gradHiddenStateShape->GetDim(DIM_BATCH) != B ||
        irShape->GetDim(DIM_BATCH) != B || pbShape->GetDim(DIM_BATCH) != B) {
        return false;
    }
    if (brShape->GetDim(DIM_BLOCK_RES_HIDDEN) != H || pwShape->GetDim(DIM_WEIGHT_ROW) != 1 ||
        pwShape->GetDim(DIM_HIDDEN) != H || nwShape->GetDim(DIM_NORM_WEIGHT) != H ||
        gradHiddenStateShape->GetDim(DIM_HIDDEN) != H) {
        return false;
    }
    if (irShape->GetDim(DIM_BLOCK) != N + 1 || pbShape->GetDim(DIM_BLOCK) != N + 1) {
        return false;
    }

    return true;
}

static ge::graphStatus InferShapeBlockAttentionResidualsGrad(gert::InferShapeContext *context)
{
    int64_t B = 0, N = 0, H = 0;
    if (!CheckShapeBlockAttentionResidualsGrad(context, B, N, H)) {
        return GRAPH_FAILED;
    }

    // grad_partial_block [B, H] = partial_block
    context->GetOutputShape(OUTPUT_GRAD_PARTIAL_BLOCK)->SetDimNum(RANK_2D);
    context->GetOutputShape(OUTPUT_GRAD_PARTIAL_BLOCK)->SetDim(DIM_BATCH, B);
    context->GetOutputShape(OUTPUT_GRAD_PARTIAL_BLOCK)->SetDim(DIM_HIDDEN, H);

    // grad_block_res [B, N, H] = block_res
    context->GetOutputShape(OUTPUT_GRAD_BLOCK_RES)->SetDimNum(RANK_3D);
    context->GetOutputShape(OUTPUT_GRAD_BLOCK_RES)->SetDim(DIM_BATCH, B);
    context->GetOutputShape(OUTPUT_GRAD_BLOCK_RES)->SetDim(DIM_BLOCK, N);
    context->GetOutputShape(OUTPUT_GRAD_BLOCK_RES)->SetDim(DIM_BLOCK_RES_HIDDEN, H);

    // grad_proj_weight [H] = proj_weight
    context->GetOutputShape(OUTPUT_GRAD_PROJ_WEIGHT)->SetDimNum(RANK_2D);
    context->GetOutputShape(OUTPUT_GRAD_PROJ_WEIGHT)->SetDim(DIM_WEIGHT_ROW, 1);
    context->GetOutputShape(OUTPUT_GRAD_PROJ_WEIGHT)->SetDim(DIM_HIDDEN, H);

    // grad_norm_weight [H] = norm_weight
    context->GetOutputShape(OUTPUT_GRAD_NORM_WEIGHT)->SetDimNum(RANK_1D);
    context->GetOutputShape(OUTPUT_GRAD_NORM_WEIGHT)->SetDim(DIM_NORM_WEIGHT, H);

    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeBlockAttentionResidualsGrad(gert::InferDataTypeContext *context)
{
    const auto invNormDtype = context->GetInputDataType(INPUT_INV_NORM);
    const auto probsDtype = context->GetInputDataType(INPUT_PROBS);
    if (invNormDtype != ge::DT_FLOAT || probsDtype != ge::DT_FLOAT) {
        OP_LOGE(context->GetNodeName(), "inv_norm and probs only support fp32, got %d and %d",
                static_cast<int>(invNormDtype), static_cast<int>(probsDtype));
        return GRAPH_FAILED;
    }

    auto dtype = context->GetInputDataType(INPUT_PARTIAL_BLOCK);
    context->SetOutputDataType(OUTPUT_GRAD_PARTIAL_BLOCK, dtype);
    context->SetOutputDataType(OUTPUT_GRAD_BLOCK_RES, dtype);
    context->SetOutputDataType(OUTPUT_GRAD_PROJ_WEIGHT, dtype);
    context->SetOutputDataType(OUTPUT_GRAD_NORM_WEIGHT, dtype);

    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(BlockAttentionResidualsGrad)
    .InferShape(InferShapeBlockAttentionResidualsGrad)
    .InferDataType(InferDataTypeBlockAttentionResidualsGrad);
} // namespace ops
