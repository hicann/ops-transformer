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
#include <string>

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
constexpr int64_t UNKNOWN_DIM = -1;   // unknown shape dim
constexpr int64_t UNKNOWN_RANK = -2;  // unknown rank marker (dimNum=1, dim0=-2)
} // namespace

static bool IsUnknownRankShape(const gert::Shape *shape)
{
    return shape != nullptr && shape->GetDimNum() == RANK_1D && shape->GetDim(0) == UNKNOWN_RANK;
}

// 从候选输入中依次取第一个已知维度值；全部为 -1 时返回 UNKNOWN_DIM。
static int64_t ResolveFirstKnownDim(const gert::Shape *const shapes[], const size_t dims[], size_t shapeNum)
{
    size_t shapeIndex = 0;
    while (shapeIndex < shapeNum) {
        const int64_t value = shapes[shapeIndex]->GetDim(dims[shapeIndex]);
        if (value != UNKNOWN_DIM) {
            return value;
        }
        ++shapeIndex;
    }
    return UNKNOWN_DIM;
}

// 取有效 N：block_res 为 -1 时用 inv_norm/probs 的 K - 1 反推；全部未知时保持 -1。
static int64_t ResolveNumBlocks(const gert::Shape *blockResShape, const gert::Shape *invNormShape,
                                const gert::Shape *probsShape)
{
    const int64_t numBlocks = blockResShape->GetDim(DIM_BLOCK);
    if (numBlocks != UNKNOWN_DIM) {
        return numBlocks;
    }
    const int64_t invNormBlocks = invNormShape->GetDim(DIM_BLOCK);
    if (invNormBlocks != UNKNOWN_DIM) {
        return invNormBlocks - 1;
    }
    const int64_t probsBlocks = probsShape->GetDim(DIM_BLOCK);
    if (probsBlocks != UNKNOWN_DIM) {
        return probsBlocks - 1;
    }
    return UNKNOWN_DIM;
}

static bool CheckInputRank(gert::InferShapeContext *context, const gert::Shape *shape, size_t expectedRank,
                           const char *tensorName)
{
    if (shape->GetDimNum() == expectedRank) {
        return true;
    }
    OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), tensorName, std::to_string(shape->GetDimNum()).c_str(),
                                 std::to_string(expectedRank).c_str());
    return false;
}

static bool CheckInputDimValue(gert::InferShapeContext *context, const gert::Shape *shape, size_t dimIndex,
                               int64_t expectedValue, const char *dimName)
{
    const int64_t actualValue = shape->GetDim(dimIndex);
    if (actualValue == UNKNOWN_DIM || expectedValue == UNKNOWN_DIM || actualValue == expectedValue) {
        return true;
    }
    const std::string reason = std::string(dimName) + " must be " + std::to_string(expectedValue);
    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), dimName, std::to_string(actualValue).c_str(),
                                          reason.c_str());
    return false;
}

static bool IsSupportedMainDtype(ge::DataType dtype)
{
    return dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16 || dtype == ge::DT_FLOAT;
}

static const char *DtypeName(ge::DataType dataType)
{
    switch (dataType) {
        case ge::DT_FLOAT16:
            return "FLOAT16";
        case ge::DT_BF16:
            return "BF16";
        case ge::DT_FLOAT:
            return "FLOAT32";
        default:
            return "UNKNOWN";
    }
}

static ge::graphStatus CheckSupportedMainDtype(gert::InferDataTypeContext *context, const char *tensorName,
                                               ge::DataType dtype)
{
    if (IsSupportedMainDtype(dtype)) {
        return GRAPH_SUCCESS;
    }
    OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), tensorName, DtypeName(dtype), "FLOAT16/BFLOAT16/FLOAT32");
    return GRAPH_FAILED;
}

static ge::graphStatus CheckMainDtypeSame(gert::InferDataTypeContext *context, const char *tensorName,
                                          ge::DataType dtype, ge::DataType referenceDtype)
{
    if (dtype == referenceDtype) {
        return GRAPH_SUCCESS;
    }
    OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(context->GetNodeName(), tensorName, DtypeName(dtype),
                                          "main input dtype must be same as partial_block");
    return GRAPH_FAILED;
}

static bool CheckShapeBlockAttentionResidualsGrad(gert::InferShapeContext *context, int64_t &B, int64_t &N, int64_t &H)
{
    const gert::Shape *partialBlockShape = context->GetInputShape(INPUT_PARTIAL_BLOCK);
    const gert::Shape *blockResShape = context->GetInputShape(INPUT_BLOCK_RES);
    const gert::Shape *projWeightShape = context->GetInputShape(INPUT_PROJ_WEIGHT);
    const gert::Shape *normWeightShape = context->GetInputShape(INPUT_NORM_WEIGHT);
    const gert::Shape *gradHiddenStatesShape = context->GetInputShape(INPUT_GRAD_HIDDEN_STATES);
    const gert::Shape *invNormShape = context->GetInputShape(INPUT_INV_NORM);
    const gert::Shape *probsShape = context->GetInputShape(INPUT_PROBS);

    if (!CheckInputRank(context, partialBlockShape, RANK_2D, "partial_block") ||
        !CheckInputRank(context, blockResShape, RANK_3D, "block_res") ||
        !CheckInputRank(context, projWeightShape, RANK_2D, "proj_weight") ||
        !CheckInputRank(context, normWeightShape, RANK_1D, "norm_weight") ||
        !CheckInputRank(context, gradHiddenStatesShape, RANK_2D, "grad_hidden_states") ||
        !CheckInputRank(context, invNormShape, RANK_2D, "inv_norm") ||
        !CheckInputRank(context, probsShape, RANK_2D, "probs")) {
        return false;
    }

    // 先取有效 B：partial_block 为 -1 时逐个从其它输入取已知值；全部未知时保持 -1。
    const gert::Shape *batchShapes[] = {partialBlockShape, blockResShape, gradHiddenStatesShape, invNormShape,
                                        probsShape};
    const size_t batchDims[] = {DIM_BATCH, DIM_BATCH, DIM_BATCH, DIM_BATCH, DIM_BATCH};
    B = ResolveFirstKnownDim(batchShapes, batchDims, sizeof(batchShapes) / sizeof(batchShapes[0]));
    // 逐输入校验 batch 维；未知维（-1）自动跳过比较。
    if (!CheckInputDimValue(context, blockResShape, DIM_BATCH, B, "block_res.shape[0]") ||
        !CheckInputDimValue(context, gradHiddenStatesShape, DIM_BATCH, B, "grad_hidden_states.shape[0]") ||
        !CheckInputDimValue(context, invNormShape, DIM_BATCH, B, "inv_norm.shape[0]") ||
        !CheckInputDimValue(context, probsShape, DIM_BATCH, B, "probs.shape[0]")) {
        return false;
    }

    // 先取有效 H：partial_block 为 -1 时逐个从其它输入取已知值；全部未知时保持 -1。
    const gert::Shape *hiddenShapes[] = {partialBlockShape, blockResShape, projWeightShape, normWeightShape,
                                         gradHiddenStatesShape};
    const size_t hiddenDims[] = {DIM_HIDDEN, DIM_BLOCK_RES_HIDDEN, DIM_HIDDEN, DIM_NORM_WEIGHT, DIM_HIDDEN};
    H = ResolveFirstKnownDim(hiddenShapes, hiddenDims, sizeof(hiddenShapes) / sizeof(hiddenShapes[0]));
    // 逐输入校验 hidden 维；未知维（-1）自动跳过比较。
    if (!CheckInputDimValue(context, blockResShape, DIM_BLOCK_RES_HIDDEN, H, "block_res.shape[2]") ||
        !CheckInputDimValue(context, projWeightShape, DIM_HIDDEN, H, "proj_weight.shape[1]") ||
        !CheckInputDimValue(context, normWeightShape, DIM_NORM_WEIGHT, H, "norm_weight.shape[0]") ||
        !CheckInputDimValue(context, gradHiddenStatesShape, DIM_HIDDEN, H, "grad_hidden_states.shape[1]")) {
        return false;
    }

    N = ResolveNumBlocks(blockResShape, invNormShape, probsShape);

    // block_res 的 N 与 inv_norm/probs 的 K 满足 K = N + 1；任一侧未知时不做强校验。
    if (N != UNKNOWN_DIM && !CheckInputDimValue(context, invNormShape, DIM_BLOCK, N + 1, "inv_norm.shape[1]")) {
        return false;
    }
    if (N != UNKNOWN_DIM && !CheckInputDimValue(context, probsShape, DIM_BLOCK, N + 1, "probs.shape[1]")) {
        return false;
    }

    if (!CheckInputDimValue(context, projWeightShape, DIM_WEIGHT_ROW, 1, "proj_weight.shape[0]")) {
        return false;
    }

    return true;
}

static ge::graphStatus InferShapeBlockAttentionResidualsGrad(gert::InferShapeContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("BlockAttentionResidualsGrad", "infer shape context is nullptr"),
                return GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "Enter BlockAttentionResidualsGrad InferShape.");

    const gert::Shape *partialBlockShape = context->GetInputShape(INPUT_PARTIAL_BLOCK);
    const gert::Shape *blockResShape = context->GetInputShape(INPUT_BLOCK_RES);
    const gert::Shape *projWeightShape = context->GetInputShape(INPUT_PROJ_WEIGHT);
    const gert::Shape *normWeightShape = context->GetInputShape(INPUT_NORM_WEIGHT);
    const gert::Shape *gradHiddenStatesShape = context->GetInputShape(INPUT_GRAD_HIDDEN_STATES);
    const gert::Shape *invNormShape = context->GetInputShape(INPUT_INV_NORM);
    const gert::Shape *probsShape = context->GetInputShape(INPUT_PROBS);
    OP_CHECK_NULL_WITH_CONTEXT(context, partialBlockShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, blockResShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, projWeightShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, normWeightShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradHiddenStatesShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, invNormShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, probsShape);

    gert::Shape *gradPartialBlockShape = context->GetOutputShape(OUTPUT_GRAD_PARTIAL_BLOCK);
    gert::Shape *gradBlockResShape = context->GetOutputShape(OUTPUT_GRAD_BLOCK_RES);
    gert::Shape *gradProjWeightShape = context->GetOutputShape(OUTPUT_GRAD_PROJ_WEIGHT);
    gert::Shape *gradNormWeightShape = context->GetOutputShape(OUTPUT_GRAD_NORM_WEIGHT);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradPartialBlockShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradBlockResShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradProjWeightShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, gradNormWeightShape);

    // -2 unknown rank：输出直接继承对应输入 shape。
    if (IsUnknownRankShape(partialBlockShape) || IsUnknownRankShape(blockResShape) ||
        IsUnknownRankShape(projWeightShape) || IsUnknownRankShape(normWeightShape) ||
        IsUnknownRankShape(gradHiddenStatesShape) || IsUnknownRankShape(invNormShape) ||
        IsUnknownRankShape(probsShape)) {
        OP_LOGD(context->GetNodeName(), "input contains unknown rank(-2), output copies corresponding input shape.");
        *gradPartialBlockShape = *partialBlockShape;
        *gradBlockResShape = *blockResShape;
        *gradProjWeightShape = *projWeightShape;
        *gradNormWeightShape = *normWeightShape;
        OP_LOGD(context->GetNodeName(), "Exit BlockAttentionResidualsGrad InferShape with unknown rank.");
        return GRAPH_SUCCESS;
    }

    int64_t B = 0;
    int64_t N = 0;
    int64_t H = 0;
    if (!CheckShapeBlockAttentionResidualsGrad(context, B, N, H)) {
        return GRAPH_FAILED;
    }

    gradPartialBlockShape->SetDimNum(RANK_2D);
    gradPartialBlockShape->SetDim(DIM_BATCH, B);
    gradPartialBlockShape->SetDim(DIM_HIDDEN, H);

    gradBlockResShape->SetDimNum(RANK_3D);
    gradBlockResShape->SetDim(DIM_BATCH, B);
    gradBlockResShape->SetDim(DIM_BLOCK, N);
    gradBlockResShape->SetDim(DIM_BLOCK_RES_HIDDEN, H);

    gradProjWeightShape->SetDimNum(RANK_2D);
    gradProjWeightShape->SetDim(DIM_WEIGHT_ROW, 1);
    gradProjWeightShape->SetDim(DIM_HIDDEN, H);

    gradNormWeightShape->SetDimNum(RANK_1D);
    gradNormWeightShape->SetDim(DIM_NORM_WEIGHT, H);

    OP_LOGD(context->GetNodeName(), "Exit BlockAttentionResidualsGrad InferShape: B=%ld N=%ld H=%ld.", B, N, H);
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeBlockAttentionResidualsGrad(gert::InferDataTypeContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("BlockAttentionResidualsGrad", "infer data type context is nullptr"),
                return GRAPH_FAILED);
    OP_LOGD(context->GetNodeName(), "Enter BlockAttentionResidualsGrad InferDataType.");

    const auto partialBlockDtype = context->GetInputDataType(INPUT_PARTIAL_BLOCK);
    const auto blockResDtype = context->GetInputDataType(INPUT_BLOCK_RES);
    const auto projWeightDtype = context->GetInputDataType(INPUT_PROJ_WEIGHT);
    const auto normWeightDtype = context->GetInputDataType(INPUT_NORM_WEIGHT);
    const auto gradHiddenStatesDtype = context->GetInputDataType(INPUT_GRAD_HIDDEN_STATES);
    const auto invNormDtype = context->GetInputDataType(INPUT_INV_NORM);
    const auto probsDtype = context->GetInputDataType(INPUT_PROBS);

    if (CheckSupportedMainDtype(context, "partial_block", partialBlockDtype) != GRAPH_SUCCESS ||
        CheckSupportedMainDtype(context, "block_res", blockResDtype) != GRAPH_SUCCESS ||
        CheckSupportedMainDtype(context, "proj_weight", projWeightDtype) != GRAPH_SUCCESS ||
        CheckSupportedMainDtype(context, "norm_weight", normWeightDtype) != GRAPH_SUCCESS ||
        CheckSupportedMainDtype(context, "grad_hidden_states", gradHiddenStatesDtype) != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
    }
    if (CheckMainDtypeSame(context, "block_res", blockResDtype, partialBlockDtype) != GRAPH_SUCCESS ||
        CheckMainDtypeSame(context, "proj_weight", projWeightDtype, partialBlockDtype) != GRAPH_SUCCESS ||
        CheckMainDtypeSame(context, "norm_weight", normWeightDtype, partialBlockDtype) != GRAPH_SUCCESS ||
        CheckMainDtypeSame(context, "grad_hidden_states", gradHiddenStatesDtype, partialBlockDtype) != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
    }
    if (invNormDtype != ge::DT_FLOAT) {
        OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "inv_norm", DtypeName(invNormDtype), "FLOAT32");
        return GRAPH_FAILED;
    }
    if (probsDtype != ge::DT_FLOAT) {
        OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "probs", DtypeName(probsDtype), "FLOAT32");
        return GRAPH_FAILED;
    }

    context->SetOutputDataType(OUTPUT_GRAD_PARTIAL_BLOCK, partialBlockDtype);
    context->SetOutputDataType(OUTPUT_GRAD_BLOCK_RES, partialBlockDtype);
    context->SetOutputDataType(OUTPUT_GRAD_PROJ_WEIGHT, partialBlockDtype);
    context->SetOutputDataType(OUTPUT_GRAD_NORM_WEIGHT, partialBlockDtype);

    OP_LOGD(context->GetNodeName(), "Exit BlockAttentionResidualsGrad InferDataType.");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(BlockAttentionResidualsGrad)
    .InferShape(InferShapeBlockAttentionResidualsGrad)
    .InferDataType(InferDataTypeBlockAttentionResidualsGrad);
} // namespace ops
