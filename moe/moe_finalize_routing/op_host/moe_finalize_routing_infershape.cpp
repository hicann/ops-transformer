/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_finalize_routing_ops.cc
 * \brief
 */
#include "graph/utils/type_utils.h"
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;
namespace ops {
static const size_t INDEX_IN_EXPAND_PERMUTED_ROWS = 0;
static const size_t INDEX_IN_SKIP1 = 1;
static const size_t INDEX_IN_SKIP2 = 2;
static const size_t INDEX_IN_BIAS = 3;
static const size_t INDEX_IN_SCALES = 4;
static const size_t INDEX_IN_EXPANDED_SRC_TO_DST_ROW = 5;
static const size_t INDEX_IN_EXPERT_FOR_SOURCE_ROW = 6;
static constexpr size_t INDEX_out = 0;
static constexpr size_t SHAPE_SIZE = 2;
static constexpr size_t INPUT_NUM = 7;

static inline bool IsValidType(const DataType dtype)
{
    return dtype == ge::DT_FLOAT || dtype == ge::DT_FLOAT16 || dtype == ge::DT_BF16;
}

static ge::graphStatus InferDataTypeMoeFinalizeRouting(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do MoeFinalizeRoutingInferDataType.");
    OP_CHECK_IF(
        !IsValidType(context->GetInputDataType(INDEX_IN_EXPAND_PERMUTED_ROWS)),
        OP_LOGE(context->GetNodeName(), "The dtype of expanded_permuted_rows should be float, float16 or bf16."),
        return ge::GRAPH_FAILED);

    OP_CHECK_IF(!IsValidType(context->GetInputDataType(INDEX_IN_SKIP1)),
                OP_LOGE(context->GetNodeName(), "The dtype of skip1 should be float, float16 or bf16."),
                return ge::GRAPH_FAILED);

    const DataType skip2Dtype = context->GetOptionalInputDataType(INDEX_IN_SKIP2);

    OP_CHECK_IF(skip2Dtype != ge::DT_UNDEFINED && !IsValidType(skip2Dtype),
                OP_LOGE(context->GetNodeName(), "The dtype of skip2 should be float, float16 or bf16."),
                return ge::GRAPH_FAILED);

    size_t offset = (context->GetComputeNodeInputNum() == INPUT_NUM) ? 0 : 1;

    OP_CHECK_IF(!IsValidType(context->GetInputDataType(INDEX_IN_BIAS - offset)),
                OP_LOGE(context->GetNodeName(), "The dtype of bias should be float, float16 or bf16."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(!IsValidType(context->GetInputDataType(INDEX_IN_SCALES - offset)),
                OP_LOGE(context->GetNodeName(), "The dtype of scales should be float, float16 or bf16."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(context->GetInputDataType(INDEX_IN_EXPANDED_SRC_TO_DST_ROW - offset) != ge::DT_INT32,
                OP_LOGE(context->GetNodeName(), "The dtype of expanded_src_to_dst_row should be int32."),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(context->GetInputDataType(INDEX_IN_EXPERT_FOR_SOURCE_ROW - offset) != ge::DT_INT32,
                OP_LOGE(context->GetNodeName(), "The dtype of expert_for_source_row should be int32."),
                return ge::GRAPH_FAILED);

    context->SetOutputDataType(0, context->GetInputDataType(INDEX_IN_SKIP1));
    return ge::GRAPH_SUCCESS;
}

static bool IsValidShape(const int64_t shape1, const int64_t shape2, const int64_t shape3, const int64_t shape4)
{
    std::vector<int64_t> validValue = {-1};
    std::vector<int64_t> currentValue = {shape1, shape2, shape3, shape4};
    for (auto value : currentValue) {
        if (value == -1) {
            continue;
        }
        if (validValue.size() == 1) {
            validValue.push_back(value);
        } else if (validValue[1] != value) {
            return false;
        }
    }
    return true;
}

static ge::graphStatus MoeCopyShapeInput2OutputWithIdx(gert::InferShapeContext* context, int64_t input_idx,
                                                       int64_t output_idx)
{
    auto in_shape = context->GetInputShape(input_idx);
    OP_CHECK_NULL_WITH_CONTEXT(context, in_shape);
    auto out_shape = context->GetOutputShape(output_idx);
    OP_CHECK_NULL_WITH_CONTEXT(context, out_shape);
    *out_shape = *in_shape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckInputShapes(const gert::InferShapeContext* context, const size_t offset)
{
    const gert::Shape* expandedPermutedRowsInputShape = context->GetInputShape(INDEX_IN_EXPAND_PERMUTED_ROWS);
    OP_CHECK_NULL_WITH_CONTEXT(context, expandedPermutedRowsInputShape);
    OP_CHECK_IF(expandedPermutedRowsInputShape->GetDimNum() != SHAPE_SIZE,
                OP_LOGE(context->GetNodeName(), "The expanded_permuted_rows of input should be 2D tensor."),
                return ge::GRAPH_FAILED);

    const gert::Shape* skip1InputShape = context->GetInputShape(INDEX_IN_SKIP1);
    OP_CHECK_NULL_WITH_CONTEXT(context, skip1InputShape);
    OP_CHECK_IF(skip1InputShape->GetDimNum() != SHAPE_SIZE,
                OP_LOGE(context->GetNodeName(), "The skip1 of input should be 2D tensor."), return ge::GRAPH_FAILED);

    const gert::Tensor* x2Tensor = context->GetOptionalInputTensor(INDEX_IN_SKIP2);
    const gert::Shape* skip2InputShape = (x2Tensor == nullptr || x2Tensor->GetShapeSize() == 0) ?
                                             nullptr :
                                             context->GetOptionalInputShape(INDEX_IN_SKIP2);
    OP_CHECK_IF(skip2InputShape != nullptr && skip2InputShape->GetDimNum() != SHAPE_SIZE,
                OP_LOGE(context->GetNodeName(), "The skip2 of input should be 2D tensor."), return ge::GRAPH_FAILED);

    const gert::Shape* biasInputShape = context->GetInputShape(INDEX_IN_BIAS - offset);
    OP_CHECK_NULL_WITH_CONTEXT(context, biasInputShape);
    OP_CHECK_IF(biasInputShape->GetDimNum() != SHAPE_SIZE,
                OP_LOGE(context->GetNodeName(), "The bias of input should be 2D tensor."), return ge::GRAPH_FAILED);

    const gert::Shape* scalesInputShape = context->GetInputShape(INDEX_IN_SCALES - offset);
    OP_CHECK_NULL_WITH_CONTEXT(context, scalesInputShape);
    OP_CHECK_IF(scalesInputShape->GetDimNum() != SHAPE_SIZE,
                OP_LOGE(context->GetNodeName(), "The scales of input should be 2D tensor."), return ge::GRAPH_FAILED);

    const gert::Shape* expandedSrcToDstRowInputShape =
        context->GetInputShape(INDEX_IN_EXPANDED_SRC_TO_DST_ROW - offset);
    OP_CHECK_NULL_WITH_CONTEXT(context, expandedSrcToDstRowInputShape);
    OP_CHECK_IF(expandedSrcToDstRowInputShape->GetDimNum() != 1,
                OP_LOGE(context->GetNodeName(), "The expanded_src_to_dst_row of input should be 1D tensor."),
                return ge::GRAPH_FAILED);

    const gert::Shape* expertForSourceRowInputShape = context->GetInputShape(INDEX_IN_EXPERT_FOR_SOURCE_ROW - offset);
    OP_CHECK_NULL_WITH_CONTEXT(context, expertForSourceRowInputShape);
    OP_CHECK_IF(expertForSourceRowInputShape->GetDimNum() != SHAPE_SIZE,
                OP_LOGE(context->GetNodeName(), "The expert_for_source_row of input should be 2D tensor."),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckInputDims(const gert::InferShapeContext* context, const size_t offset)
{
    const gert::Shape* scalesInputShape = context->GetInputShape(INDEX_IN_SCALES - offset);
    const gert::Shape* expertForSourceRowInputShape = context->GetInputShape(INDEX_IN_EXPERT_FOR_SOURCE_ROW - offset);
    bool validColK = scalesInputShape->GetDim(1) == -1 || expertForSourceRowInputShape->GetDim(1) == -1 ||
                     scalesInputShape->GetDim(1) == expertForSourceRowInputShape->GetDim(1);
    OP_CHECK_IF(!validColK,
                OP_LOGE(context->GetNodeName(), "The dim 1 of scales and expert_for_source_row should be same."),
                return ge::GRAPH_FAILED);

    const gert::Shape* skip1InputShape = context->GetInputShape(INDEX_IN_SKIP1);
    const gert::Shape* skip2InputShape = (context->GetOptionalInputTensor(INDEX_IN_SKIP2) == nullptr ||
                                          context->GetOptionalInputTensor(INDEX_IN_SKIP2)->GetShapeSize() == 0) ?
                                             nullptr :
                                             context->GetOptionalInputShape(INDEX_IN_SKIP2);
    int64_t skip2Row = skip2InputShape != nullptr ? skip2InputShape->GetDim(0) : -1;
    OP_CHECK_IF(
        !IsValidShape(skip1InputShape->GetDim(0), skip2Row, scalesInputShape->GetDim(0),
                      expertForSourceRowInputShape->GetDim(0)),
        OP_LOGE(context->GetNodeName(), "The dim 0 of skip1, skip2, scales and expert_for_source_row should be same."),
        return ge::GRAPH_FAILED);

    int64_t skip2Col = skip2InputShape != nullptr ? skip2InputShape->GetDim(1) : -1;
    OP_CHECK_IF(
        !IsValidShape(skip1InputShape->GetDim(1), skip2Col,
                      context->GetInputShape(INDEX_IN_EXPAND_PERMUTED_ROWS)->GetDim(1),
                      context->GetInputShape(INDEX_IN_BIAS - offset)->GetDim(1)),
        OP_LOGE(context->GetNodeName(), "The dim 1 of skip1, skip2, expanded_permuted_rows and bias should be same."),
        return ge::GRAPH_FAILED);

    const gert::Shape* expandedSrcToDstRowInputShape =
        context->GetInputShape(INDEX_IN_EXPANDED_SRC_TO_DST_ROW - offset);
    const gert::Shape* expandedPermutedRowsInputShape = context->GetInputShape(INDEX_IN_EXPAND_PERMUTED_ROWS);
    bool validDim = expandedSrcToDstRowInputShape->GetDim(0) == -1 || expandedPermutedRowsInputShape->GetDim(0) == -1 ||
                    expandedSrcToDstRowInputShape->GetDim(0) == expandedPermutedRowsInputShape->GetDim(0);

    OP_CHECK_IF(!validDim,
                OP_LOGE(context->GetNodeName(),
                        "The dim 0 of expanded_permuted_rows and expanded_src_to_dst_row should be same."),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Infershape4MoeFinalizeRouting(gert::InferShapeContext* context)
{
    // get and check input param
    size_t offset = (context->GetComputeNodeInputNum() == INPUT_NUM) ? 0 : 1;
    if (CheckInputShapes(context, offset) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (CheckInputDims(context, offset) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // infershape output
    OP_CHECK_IF(MoeCopyShapeInput2OutputWithIdx(context, INDEX_IN_SKIP1, INDEX_out) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "Infershape4MoeFinalizeRouting failed!"), return ge::GRAPH_FAILED);

    OP_LOGD(context->GetNodeName(), "End to do MoeFinalizeRoutingInfershape.");

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MoeFinalizeRouting)
    .InferShape(Infershape4MoeFinalizeRouting)
    .InferDataType(InferDataTypeMoeFinalizeRouting);
} // namespace ops
