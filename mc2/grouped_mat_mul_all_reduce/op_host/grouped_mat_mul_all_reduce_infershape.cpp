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
 * \file grouped_mat_mul_all_reduce_infershape.cpp
 * \brief
 */
#include "mc2_log.h"
#include "register/op_impl_registry.h"
#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/infer_datatype_context.h"

using namespace ge;
namespace ops {

static const size_t INDEX_IN_X = 0;
static const size_t INDEX_IN_WEIGHT = 1;
static const size_t INDEX_IN_BIAS = 2;
static const size_t INDEX_IN_GROUP_LIST = 3;
static const size_t INDEX_IN_CONTEXT = 4;
static const size_t INDEX_OUT_Y = 0;
static const size_t INDEX_ATTR_SPLIT_ITEM = 0;

static const int32_t IN_NOT_SPLIT_OUT_NOT_SPLIT = 0;
static const int32_t IN_SPLIT_OUT_NOT_SPLIT = 1;
static const int32_t IN_NOT_SPLIT_OUT_SPLIT = 2;
static const int32_t IN_SPLIT_OUT_SPLIT = 3;

static ge::graphStatus CheckSplitItemGmmAr(int32_t splitItem)
{
    if (splitItem == IN_NOT_SPLIT_OUT_NOT_SPLIT || splitItem == IN_SPLIT_OUT_SPLIT ||
        splitItem == IN_NOT_SPLIT_OUT_SPLIT || splitItem == IN_SPLIT_OUT_NOT_SPLIT) {
        return GRAPH_SUCCESS;
    } else {
        return GRAPH_FAILED;
    }
}

static graphStatus CheckDims(const gert::InferShapeContext *context, const gert::Shape *xShape,
                             const gert::Shape *weightShape, int32_t splitItem)
{
    auto result = GRAPH_SUCCESS;

    auto dimNumX = xShape->GetDimNum();
    auto dimNumWeight = weightShape->GetDimNum();
    if (splitItem == 0 && ((dimNumX < 2 || dimNumX > 6))) { // 2 is minDimNumX, 6 is maxDimNumX
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(),
                                            "When splitItem = 0, only x with dim 2-6 is supported.");
        result = GRAPH_FAILED;
    } else if (splitItem != 0 && dimNumX != 2) { // 2 is minDimNumX
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(),
                                            "When splitItem != 0, only x with dim 2 is supported.");
        result = GRAPH_FAILED;
    }
    if (dimNumWeight != 2) { // 2 is required dimNumWeight
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), "Only weight with dim 2 is supported.");
        result = GRAPH_FAILED;
    }

    auto lastDimX = xShape->GetDim(xShape->GetDimNum() - 1);
    auto firstDimWeight = weightShape->GetDim(0);
    if (lastDimX != firstDimWeight) {
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(),
                                            "For each matmul, the last dim of x must equal the first dim of weight.");
        result = GRAPH_FAILED;
    }

    return result;
}

static graphStatus UpdateShapeYMultiDimGMMAllReduce(gert::InferShapeContext *context, size_t yIndex,
                                                    const gert::Shape *xShape, const gert::Shape *weightShape)
{
    gert::Shape *yShape = context->GetOutputShape(yIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    *yShape = *xShape;
    size_t dimY = yShape->GetDimNum();
    yShape->SetDim(dimY - 1, weightShape->GetDim(1));
    return GRAPH_SUCCESS;
}

static graphStatus UpdateShapeYGMMAllReduce(gert::InferShapeContext *context, size_t yIndex, int64_t dim0, int64_t dim1)
{
    gert::Shape *yShape = context->GetOutputShape(yIndex);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    auto xShape = context->GetInputShape(INDEX_IN_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    yShape->SetDimNum(xShape->GetDimNum());
    yShape->SetDim(0, dim0);
    yShape->SetDim(1, dim1);
    return GRAPH_SUCCESS;
}

static graphStatus CheckSplitItem0(gert::InferShapeContext *context, size_t &idx)
{
    while (1) {
        const gert::Shape *xShape = context->GetDynamicInputShape(INDEX_IN_X, idx);
        if (xShape == nullptr) {
            break;
        }
        const gert::Shape *weightShape = context->GetDynamicInputShape(INDEX_IN_WEIGHT, idx);
        OP_CHECK_NULL_WITH_CONTEXT(context, weightShape);
        OPS_CHECK(CheckDims(context, xShape, weightShape, 0) == GRAPH_FAILED,
                  VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), "Invalid dims of x or weight."),
                  return GRAPH_FAILED);
        OPS_CHECK(UpdateShapeYMultiDimGMMAllReduce(context, INDEX_OUT_Y + idx, xShape, weightShape) != GRAPH_SUCCESS,
                  VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), "Failed to update shape of y."),
                  return GRAPH_FAILED);
        idx++;
    }
    return GRAPH_SUCCESS;
}

static graphStatus CheckSplitItem1(gert::InferShapeContext *context, size_t &idx)
{
    const gert::Tensor *groupListTensor = context->GetInputTensor(context->GetComputeNodeInputNum());
    OP_CHECK_NULL_WITH_CONTEXT(context, groupListTensor);
    const gert::Shape *xShape = context->GetDynamicInputShape(INDEX_IN_X, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    int64_t preOffset = 0;
    while (1) {
        const gert::Shape *weightShape = context->GetDynamicInputShape(INDEX_IN_WEIGHT, idx);
        if (weightShape == nullptr) {
            break;
        }
        OPS_CHECK(CheckDims(context, xShape, weightShape, 1) == GRAPH_FAILED,
                  VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), "Invalid dims of x or weight."),
                  return GRAPH_FAILED);
        auto data = groupListTensor->GetData<int64_t>();
        OP_CHECK_NULL_WITH_CONTEXT(context, data);
        OPS_CHECK(data[idx] <= preOffset,
                  VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(),
                                                      "group_list should be strictly monotonically increasing."),
                  return GRAPH_FAILED);
        OPS_CHECK(UpdateShapeYGMMAllReduce(context, INDEX_OUT_Y + idx, data[idx] - preOffset, weightShape->GetDim(1)) !=
                      GRAPH_SUCCESS,
                  VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), "Failed to update shape of y."),
                  return GRAPH_FAILED);
        preOffset = data[idx];
        idx++;
    }
    return GRAPH_SUCCESS;
}

static graphStatus InferMAxisShape(gert::InferShapeContext *context)
{
    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int32_t *splitItem = attrs->GetAttrPointer<int32_t>(INDEX_ATTR_SPLIT_ITEM);
    OP_CHECK_NULL_WITH_CONTEXT(context, splitItem);
    OPS_CHECK(
        CheckSplitItemGmmAr(*splitItem) != GRAPH_SUCCESS,
        OP_LOGE_WITH_INVALID_ATTR(context->GetNodeName(), "splitItem", std::to_string(*splitItem).c_str(), "0/1/2/3"),
        return GRAPH_FAILED);
    auto w0Shape = context->GetDynamicInputShape(INDEX_IN_WEIGHT, 0);
    OP_CHECK_NULL_WITH_CONTEXT(context, w0Shape);
    size_t idx = 0;
    if (*splitItem == IN_NOT_SPLIT_OUT_NOT_SPLIT) {
        OPS_CHECK(CheckSplitItem0(context, idx) == GRAPH_FAILED,
                  VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), "CheckSplitItem0 failed"),
                  return GRAPH_FAILED);
    } else if (*splitItem == IN_SPLIT_OUT_NOT_SPLIT) {
        OPS_CHECK(CheckSplitItem1(context, idx) == GRAPH_FAILED,
                  VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), "CheckSplitItem1 failed"),
                  return GRAPH_FAILED);
    } else if (*splitItem == IN_NOT_SPLIT_OUT_SPLIT) {
        int64_t m = 0;
        while (1) {
            const gert::Shape *xShape = context->GetDynamicInputShape(INDEX_IN_X, idx);
            if (xShape == nullptr) {
                break;
            }
            const gert::Shape *weightShape = context->GetDynamicInputShape(INDEX_IN_WEIGHT, idx);
            OP_CHECK_NULL_WITH_CONTEXT(context, weightShape);
            OPS_CHECK(CheckDims(context, xShape, weightShape, *splitItem) == GRAPH_FAILED,
                      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), "Invalid dims of x or weight."),
                      return GRAPH_FAILED);
            m += xShape->GetDim(0);
            idx++;
        }
        OPS_CHECK(UpdateShapeYGMMAllReduce(context, INDEX_OUT_Y, m, w0Shape->GetDim(1)) != GRAPH_SUCCESS,
                  VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), "Failed to update shape of y."),
                  return GRAPH_FAILED);
    } else if (*splitItem == IN_SPLIT_OUT_SPLIT) {
        const gert::Shape *xShape = context->GetDynamicInputShape(INDEX_IN_X, 0);
        OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
        while (1) {
            const gert::Shape *weightShape = context->GetDynamicInputShape(INDEX_IN_WEIGHT, idx);
            if (weightShape == nullptr) {
                break;
            }
            OPS_CHECK(CheckDims(context, xShape, weightShape, *splitItem) == GRAPH_FAILED,
                      VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), "Invalid dims of x or weight."),
                      return GRAPH_FAILED);
            idx++;
        }
        OPS_CHECK(
            UpdateShapeYGMMAllReduce(context, INDEX_OUT_Y, xShape->GetDim(0), w0Shape->GetDim(1)) != GRAPH_SUCCESS,
            VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), "Failed to update shape of y."),
            return GRAPH_FAILED);
    }
    return GRAPH_SUCCESS;
}

static graphStatus InferDataTypeGroupedMatMulAllReduce(gert::InferDataTypeContext *context)
{
    auto x0Dtype = context->GetDynamicInputDataType(INDEX_IN_X, 0);
    auto w0Dtype = context->GetDynamicInputDataType(INDEX_IN_WEIGHT, 0);
    size_t idxX = 1;
    while (1) {
        auto xDtype = context->GetDynamicInputDataType(INDEX_IN_X, idxX);
        if (DT_UNDEFINED == xDtype) {
            break;
        }
        OPS_CHECK(xDtype != x0Dtype,
                  VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(),
                                                      "Data type of input x tensors should all be the same."),
                  return GRAPH_FAILED);
        idxX++;
    }
    size_t idxWeight = 1;
    while (1) {
        auto wDtype = context->GetDynamicInputDataType(INDEX_IN_WEIGHT, idxWeight);
        if (DT_UNDEFINED == wDtype) {
            break;
        }
        OPS_CHECK(wDtype != w0Dtype,
                  VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(),
                                                      "Data type of input weight tensors should all be the same."),
                  return GRAPH_FAILED);
        idxWeight++;
    }

    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int32_t *splitItem = attrs->GetAttrPointer<int32_t>(INDEX_ATTR_SPLIT_ITEM);
    if (nullptr == splitItem) {
        return GRAPH_FAILED;
    }
    if (1 == *splitItem || 3 == *splitItem) { // group_list is required when splitItem = 1/3
        size_t numIn = context->GetComputeNodeInputNum();
        auto groupListDtype = context->GetInputDataType(numIn);
        OPS_CHECK(
            DT_INT64 != groupListDtype,
            VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), "Only int64 is supported for group_list."),
            return GRAPH_FAILED);
    }

    size_t numY = context->GetComputeNodeOutputNum();
    for (size_t k = 0; k < numY; k++) {
        context->SetOutputDataType(INDEX_OUT_Y + k, x0Dtype);
    }
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(GroupedMatMulAllReduce)
    .InferShape(InferMAxisShape)
    .InputsDataDependency({INDEX_IN_GROUP_LIST})
    .InferDataType(InferDataTypeGroupedMatMulAllReduce);
} // namespace ops
