/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file moe_init_routing_infershape.cpp
 * \brief
 */
#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;
namespace ops {
static constexpr size_t DIM_ONE = 1;
static constexpr size_t DIM_TWO = 2;
static constexpr int64_t NEG_ONE = -1;
static constexpr int64_t EXPENDED_X_IDX = 0;
static constexpr int64_t EXPENDED_ROW_IDX = 1;
static constexpr int64_t EXPENDED_EXPERT_IDX = 2;

static bool isSameDim(int64_t dim1, int64_t dim2)
{
    if (dim1 == NEG_ONE || dim2 == NEG_ONE) {
        return true;
    }
    return dim1 == dim2;
}

static ge::graphStatus CheckInputShape(const gert::InferShapeContext* context, const gert::Shape* xShape,
                                       const gert::Shape* rowIdxShape, const gert::Shape* expertIdxShape)
{
    int64_t x_n = xShape->GetDimNum() == 1U ? NEG_ONE : xShape->GetDim(0);
    int64_t cols = xShape->GetDimNum() == 1U ? NEG_ONE : xShape->GetDim(1);
    if (x_n < NEG_ONE || cols < NEG_ONE) {
        std::string xShapeStr = Ops::Base::ToString(*xShape);
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "x", xShapeStr.c_str(),
                                              "The shape of input x is invalid");
        return ge::GRAPH_FAILED;
    }

    int64_t row_idx_n = rowIdxShape->GetDimNum() == 1U ? NEG_ONE : rowIdxShape->GetDim(0);
    int64_t row_idx_k = rowIdxShape->GetDimNum() == 1U ? NEG_ONE : rowIdxShape->GetDim(1);
    if (row_idx_n < NEG_ONE || row_idx_k < NEG_ONE) {
        std::string rowIdxShapeStr = Ops::Base::ToString(*rowIdxShape);
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "row_idx", rowIdxShapeStr.c_str(),
                                              "The shape of input row_idx is invalid");
        return ge::GRAPH_FAILED;
    }

    int64_t expert_idx_n = expertIdxShape->GetDimNum() == 1U ? NEG_ONE : expertIdxShape->GetDim(0);
    int64_t expert_idx_k = expertIdxShape->GetDimNum() == 1U ? NEG_ONE : expertIdxShape->GetDim(1);
    if (expert_idx_n < NEG_ONE || expert_idx_k < NEG_ONE) {
        std::string expertIdxShapeStr = Ops::Base::ToString(*expertIdxShape);
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "expert_idx", expertIdxShapeStr.c_str(),
                                              "The shape of input expert_idx is invalid");
        return ge::GRAPH_FAILED;
    }

    if (!isSameDim(x_n, row_idx_n) || !isSameDim(x_n, expert_idx_n) || !isSameDim(row_idx_n, expert_idx_n)) {
        std::string shapeMsg = Ops::Base::ToString(*xShape) + ", " + Ops::Base::ToString(*rowIdxShape) + " and " +
                               Ops::Base::ToString(*expertIdxShape);
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "x, row_idx and expert_idx", shapeMsg.c_str(),
                                               "The first dim of x, row_idx and expert_idx should be same");
        return ge::GRAPH_FAILED;
    }

    if (!isSameDim(row_idx_k, expert_idx_k)) {
        std::string shapeMsg = Ops::Base::ToString(*rowIdxShape) + " and " + Ops::Base::ToString(*expertIdxShape);
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "row_idx and expert_idx", shapeMsg.c_str(),
                                               "The second dim of row_idx and expert_idx should be same");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckInputDimsAndAttr(const gert::InferShapeContext* context, const gert::Shape* xShape,
                                             const gert::Shape* rowIdxShape, const gert::Shape* expertIdxShape,
                                             const int64_t activeNum)
{
    if (xShape->GetDimNum() == 1U) {
        if (xShape->GetDim(0) != ge::UNKNOWN_DIM_NUM) {
            std::string xShapeStr = Ops::Base::ToString(*xShape);
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "x", xShapeStr.c_str(),
                                                  "The dynamic dim of x should be -2");
            return ge::GRAPH_FAILED;
        }
    } else if (xShape->GetDimNum() != DIM_TWO) {
        std::string xDimNumStr = std::to_string(xShape->GetDimNum());
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "x", xDimNumStr.c_str(), "2D or dynamic");
        return ge::GRAPH_FAILED;
    }

    if (rowIdxShape->GetDimNum() == 1U) {
        if (rowIdxShape->GetDim(0) != ge::UNKNOWN_DIM_NUM) {
            std::string rowIdxShapeStr = Ops::Base::ToString(*rowIdxShape);
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "row_idx", rowIdxShapeStr.c_str(),
                                                  "The dynamic dim of row_idx should be -2");
            return ge::GRAPH_FAILED;
        }
    } else if (rowIdxShape->GetDimNum() != DIM_TWO) {
        std::string rowIdxDimNumStr = std::to_string(rowIdxShape->GetDimNum());
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "row_idx", rowIdxDimNumStr.c_str(), "2D or dynamic");
        return ge::GRAPH_FAILED;
    }

    if (expertIdxShape->GetDimNum() == 1U) {
        if (expertIdxShape->GetDim(0) != ge::UNKNOWN_DIM_NUM) {
            std::string expertIdxShapeStr = Ops::Base::ToString(*expertIdxShape);
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "expert_idx", expertIdxShapeStr.c_str(),
                                                  "The dynamic dim of expert_idx should be -2");
            return ge::GRAPH_FAILED;
        }
    } else if (expertIdxShape->GetDimNum() != DIM_TWO) {
        std::string expertIdxDimNumStr = std::to_string(expertIdxShape->GetDimNum());
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "expert_idx", expertIdxDimNumStr.c_str(), "2D or dynamic");
        return ge::GRAPH_FAILED;
    }

    if (activeNum < 0) {
        std::string activeNumStr = std::to_string(activeNum);
        OP_LOGE_WITH_INVALID_ATTR(context->GetNodeName(), "active_num", activeNumStr.c_str(), "a non-negative number");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

static void ShowInputShapeInfo(const gert::InferShapeContext* context, const gert::Shape* xShape,
                               const gert::Shape* rowIdxShape, const gert::Shape* expertIdxShape,
                               const int64_t activeNum)
{
    OP_LOGD(context->GetNodeName(), "x shape is: %s.", Ops::Base::ToString(*xShape).c_str());
    OP_LOGD(context->GetNodeName(), "row_idx shape is: %s.", Ops::Base::ToString(*rowIdxShape).c_str());
    OP_LOGD(context->GetNodeName(), "expert_idx shape is: %s.", Ops::Base::ToString(*expertIdxShape).c_str());
    OP_LOGD(context->GetNodeName(), "activeNum is: %ld.", activeNum);
}

static void ShowOutputShapeInfo(const gert::InferShapeContext* context, const gert::Shape* expandedXShape,
                                const gert::Shape* expandedRowIdx, const gert::Shape* expandedExpertIdxShape)
{
    OP_LOGD(context->GetNodeName(), "expanded_x shape is: %s after infershape.",
            Ops::Base::ToString(*expandedXShape).c_str());
    OP_LOGD(context->GetNodeName(), "expanded_row_idx shape is: %s after infershape.",
            Ops::Base::ToString(*expandedRowIdx).c_str());
    OP_LOGD(context->GetNodeName(), "expanded_expert_idx shape is: %s after infershape.",
            Ops::Base::ToString(*expandedExpertIdxShape).c_str());
}

static ge::graphStatus InferShape4MoeInitRouting(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do MoeInitRoutingInfershape.");
    // 获取输入shape
    const gert::Shape* xShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    const gert::Shape* rowIdxShape = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, rowIdxShape);
    const gert::Shape* expertIdxShape = context->GetInputShape(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, expertIdxShape);
    gert::Shape* expandedXShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, expandedXShape);
    gert::Shape* expandedRowIdx = context->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, expandedRowIdx);
    gert::Shape* expandedExpertIdxShape = context->GetOutputShape(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, expandedExpertIdxShape);
    // 获取attr
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const int64_t* activeNumPtr = attrs->GetAttrPointer<int64_t>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, activeNumPtr);
    const int64_t activeNum = *activeNumPtr;
    ShowInputShapeInfo(context, xShape, rowIdxShape, expertIdxShape, activeNum);

    // 参数校验
    if (CheckInputDimsAndAttr(context, xShape, rowIdxShape, expertIdxShape, activeNum) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if (CheckInputShape(context, xShape, rowIdxShape, expertIdxShape) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    int64_t x_n = xShape->GetDimNum() == 1U ? NEG_ONE : xShape->GetDim(0);
    int64_t cols = xShape->GetDimNum() == 1U ? NEG_ONE : xShape->GetDim(1);

    int64_t row_idx_n = rowIdxShape->GetDimNum() == 1U ? NEG_ONE : rowIdxShape->GetDim(0);
    int64_t row_idx_k = rowIdxShape->GetDimNum() == 1U ? NEG_ONE : rowIdxShape->GetDim(1);

    int64_t expert_idx_n = expertIdxShape->GetDimNum() == 1U ? NEG_ONE : expertIdxShape->GetDim(0);
    int64_t expert_idx_k = expertIdxShape->GetDimNum() == 1U ? NEG_ONE : expertIdxShape->GetDim(1);

    int64_t n = x_n > row_idx_n ? (x_n > expert_idx_n ? x_n : expert_idx_n) :
                                  (row_idx_n > expert_idx_n ? row_idx_n : expert_idx_n);
    int64_t k = std::max(row_idx_k, expert_idx_k);
    int64_t outActiveNum = (n == NEG_ONE || k == NEG_ONE) ? NEG_ONE : std::min(n, activeNum) * k;
    int64_t expertForSourceRowNum = (n == NEG_ONE || k == NEG_ONE) ? NEG_ONE : n * k;

    expandedXShape->SetDimNum(DIM_TWO);
    expandedXShape->SetDim(0U, outActiveNum);
    expandedXShape->SetDim(1U, cols);

    expandedRowIdx->SetDimNum(DIM_ONE);
    expandedRowIdx->SetDim(0U, expertForSourceRowNum);

    expandedExpertIdxShape->SetDimNum(DIM_ONE);
    expandedExpertIdxShape->SetDim(0U, expertForSourceRowNum);

    ShowOutputShapeInfo(context, expandedXShape, expandedRowIdx, expandedExpertIdxShape);
    OP_LOGD(context->GetNodeName(), "End to do MoeInitRoutingInfershape.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4MoeInitRouting(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do MoeInitRoutingInferDataType.");
    auto xDtype = context->GetInputDataType(0);
    context->SetOutputDataType(EXPENDED_X_IDX, xDtype);
    context->SetOutputDataType(EXPENDED_ROW_IDX, ge::DT_INT32);
    context->SetOutputDataType(EXPENDED_EXPERT_IDX, ge::DT_INT32);
    OP_LOGD(context->GetNodeName(), "End to do MoeInitRoutingInferDataType.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MoeInitRouting).InferShape(InferShape4MoeInitRouting).InferDataType(InferDataType4MoeInitRouting);
} // namespace ops
