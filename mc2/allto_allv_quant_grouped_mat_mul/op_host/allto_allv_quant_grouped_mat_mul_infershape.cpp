/* *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */

/* !
 * \file allto_allv_quant_grouped_mat_mul_infershape.cpp
 * \brief
 */
#include "mc2_log.h"
#include "platform/platform_info.h"
#include "register/op_impl_registry.h"
#include "graph/utils/type_utils.h"
#include "util/math_util.h"

using namespace ge;
namespace ops {

using Ops::Base::CeilDiv;

// output index
static const size_t INDEX_OUT_GMM_Y = 0;
static const size_t INDEX_OUT_MM_Y = 1;
static const size_t INDEX_PERMUTE_OUT = 2;
// input index
static const size_t INDEX_IN_GMM_X = 0;
static const size_t INDEX_IN_GMM_WEIGHT = 1;
static const size_t INDEX_IN_MM_X = 6;
static const size_t INDEX_IN_MM_WEIGHT = 7;
// dim num
static constexpr size_t DIM_NUM_0 = 0;
static constexpr size_t DIM_NUM_1 = 1;
static constexpr size_t DIM_NUM_2 = 2;
static constexpr size_t DIM_NUM_3 = 3;
// attr index
static const size_t INDEX_ATTR_EP_WORLD_SIZE = 1;
static const size_t INDEX_ATTR_SEND_COUNTS = 2;
static const size_t INDEX_ATTR_RECV_COUNTS = 3;
static const size_t INDEX_ATTR_TRANS_GMM_WEIGHT_INDEX = 6;
static const size_t INDEX_ATTR_TRANS_MM_WEIGHT_INDEX = 7;
static const size_t INDEX_ATTR_PERMUTE_OUT_FLAG_INDEX = 8;
static const size_t INDEX_ATTR_Y_DTYPE_INDEX = 12;
static const size_t INDEX_ATTR_MM_DTYPE_INDEX = 13;

static constexpr size_t DIM_0 = 0;
static constexpr size_t DIM_1 = 1;
static constexpr size_t DIM_2 = 2;
static constexpr size_t UNDEFINED_VALUE = 28;

static constexpr int64_t FIRST_ELE_SIZE = -1;

static graphStatus CheckRequiredDims(const gert::InferShapeContext *context, const gert::Shape *gmmXShape,
                                     const gert::Shape *gmmWeightShape)
{
    auto result = ge::GRAPH_SUCCESS;

    auto gmmXDimNum = gmmXShape->GetDimNum();
    auto gmmWeightDimNum = gmmWeightShape->GetDimNum();
    if (gmmXDimNum != DIM_NUM_2) {
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), "Only gmmX with dim 2 is supported.");
        result = ge::GRAPH_FAILED;
    }
    if (gmmWeightDimNum != DIM_NUM_3) {
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), "Only gmmWeight with dim 3 is supported.");
        result = ge::GRAPH_FAILED;
    }
    return result;
}

static graphStatus CheckOptionalDims(const gert::InferShapeContext *context, const gert::Shape *mmXShape,
                                     const gert::Shape *mmWeightShape)
{
    auto result = ge::GRAPH_SUCCESS;
    auto mmXDimNum = mmXShape->GetDimNum();
    auto mmWeightDimNum = mmWeightShape->GetDimNum();
    if (mmXDimNum != DIM_NUM_2) {
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), "Only mmX with dim 2 is supported.");
        result = ge::GRAPH_FAILED;
    }
    if (mmWeightDimNum != DIM_NUM_2) {
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), "Only mmWeight with dim 2 is supported.");
        result = ge::GRAPH_FAILED;
    }
    return result;
}

static ge::graphStatus InferGMMOutputShape(const gert::InferShapeContext *context, gert::Shape *gmmYShape,
                                           const int64_t *epWorldSizePtr, const gert::ContinuousVector *recvCountsPtr,
                                           const gert::ContinuousVector *sendCountsPtr, const int64_t e, int64_t &a,
                                           const int64_t n1)
{
    gmmYShape->SetDimNum(DIM_NUM_2);
    gmmYShape->SetDim(DIM_0, FIRST_ELE_SIZE);
    gmmYShape->SetDim(DIM_1, FIRST_ELE_SIZE);
    if (e != FIRST_ELE_SIZE) {
        int64_t arraySize = e * (*epWorldSizePtr);
        OPS_ERR_IF(recvCountsPtr->GetSize() < static_cast<size_t>(arraySize),
                   VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(),
                                                       "recvCounts size should not be smaller than e * epWorldSize."),
                   return ge::GRAPH_FAILED);
        OPS_ERR_IF(sendCountsPtr->GetSize() < static_cast<size_t>(arraySize),
                   VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(),
                                                       "sendCounts size should not be smaller than e * epWorldSize."),
                   return ge::GRAPH_FAILED);
        OPS_ERR_IF(recvCountsPtr->GetData() == nullptr,
                   VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), "recvCounts GetData is nullptr."),
                   return ge::GRAPH_FAILED);
        for (int64_t i = 0; i < arraySize; i++) {
            a += static_cast<const int64_t *>(recvCountsPtr->GetData())[i];
        }
        gmmYShape->SetDim(DIM_0, a);
        gmmYShape->SetDim(DIM_1, n1);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferMMOutputShape(const gert::InferShapeContext *context, const gert::Shape *mmXShape,
                                          const gert::Shape *mmWeightShape, const bool *transMmWeightPtr,
                                          gert::Shape *mmYShape)
{
    OPS_ERR_IF(mmYShape == nullptr,
               VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), "the shape of output mm_y is nullptr."),
               return ge::GRAPH_FAILED);
    mmYShape->SetDimNum(DIM_NUM_0);
    if ((mmXShape != nullptr) && (mmWeightShape != nullptr) && (transMmWeightPtr != nullptr)) {
        int64_t bs = mmXShape->GetDim(DIM_0);
        mmYShape->SetDimNum(DIM_NUM_2);
        if (bs == FIRST_ELE_SIZE) {
            mmYShape->SetDim(DIM_0, FIRST_ELE_SIZE);
            mmYShape->SetDim(DIM_1, FIRST_ELE_SIZE);
        } else {
            int64_t n2 = *transMmWeightPtr ? mmWeightShape->GetDim(DIM_0) : mmWeightShape->GetDim(DIM_1);
            mmYShape->SetDim(DIM_0, bs);
            mmYShape->SetDim(DIM_1, n2);
        }
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferPermuteOutputShape(const gert::InferShapeContext *context, const bool *permuteOutFlagPtr,
                                               const int64_t e, const int64_t a, const int64_t h,
                                               gert::Shape *permuteOutShape)
{
    ge::DataType gmmXDtype = context->GetInputDesc(INDEX_IN_GMM_X)->GetDataType();
    OPS_ERR_IF(
        permuteOutShape == nullptr,
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), "the shape of output permute_out is nullptr."),
        return ge::GRAPH_FAILED);
    permuteOutShape->SetDimNum(DIM_NUM_0);
    if ((permuteOutFlagPtr != nullptr) && (*permuteOutFlagPtr == true)) {
        permuteOutShape->SetDimNum(DIM_NUM_2);
        if (e == FIRST_ELE_SIZE) {
            permuteOutShape->SetDim(DIM_0, FIRST_ELE_SIZE);
            permuteOutShape->SetDim(DIM_1, FIRST_ELE_SIZE);
        } else {
            permuteOutShape->SetDim(DIM_0, a);
            permuteOutShape->SetDim(DIM_1, h);
        }
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeAlltoAllvGroupedMatMul(gert::InferShapeContext *context)
{
    auto *gmmXShape = context->GetInputShape(INDEX_IN_GMM_X);
    auto *gmmWeightShape = context->GetInputShape(INDEX_IN_GMM_WEIGHT);
    auto *mmXShape = context->GetOptionalInputShape(INDEX_IN_MM_X);
    auto *mmWeightShape = context->GetOptionalInputShape(INDEX_IN_MM_WEIGHT);
    auto *gmmYShape = context->GetOutputShape(INDEX_OUT_GMM_Y);
    auto *mmYShape = context->GetOutputShape(INDEX_OUT_MM_Y);
    auto *permuteOutShape = context->GetOutputShape(INDEX_PERMUTE_OUT);

    auto *attrs = context->GetAttrs();
    auto *epWorldSizePtr = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_EP_WORLD_SIZE);
    auto *recvCountsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(INDEX_ATTR_RECV_COUNTS);
    auto *sendCountsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(INDEX_ATTR_SEND_COUNTS);
    auto *transGmmWeightPtr = attrs->GetAttrPointer<bool>(INDEX_ATTR_TRANS_GMM_WEIGHT_INDEX);
    auto *transMmWeightPtr = attrs->GetAttrPointer<bool>(INDEX_ATTR_TRANS_MM_WEIGHT_INDEX);
    auto *permuteOutFlagPtr = attrs->GetAttrPointer<bool>(INDEX_ATTR_PERMUTE_OUT_FLAG_INDEX);
    OPS_CHECK_NULL_WITH_CONTEXT(context, recvCountsPtr);
    OPS_CHECK_NULL_WITH_CONTEXT(context, sendCountsPtr);
    OPS_CHECK_NULL_WITH_CONTEXT(context, transGmmWeightPtr);
    OPS_CHECK_NULL_WITH_CONTEXT(context, gmmXShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context, gmmWeightShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context, gmmYShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context, epWorldSizePtr);

    OPS_ERR_IF(CheckRequiredDims(context, gmmXShape, gmmWeightShape) != ge::GRAPH_SUCCESS,
               VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), "CheckRequiredDims failed."),
               return ge::GRAPH_FAILED);

    int64_t e = gmmWeightShape->GetDim(DIM_0);
    int64_t a = 0;
    int64_t h = gmmXShape->GetDim(DIM_1);
    int64_t n1 = *transGmmWeightPtr ? gmmWeightShape->GetDim(DIM_1) : gmmWeightShape->GetDim(DIM_2);
    if (InferGMMOutputShape(context, gmmYShape, epWorldSizePtr, recvCountsPtr, sendCountsPtr, e, a, n1) !=
        ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    if ((mmXShape != nullptr) && (mmWeightShape != nullptr)) {
        OPS_CHECK_NULL_WITH_CONTEXT(context, transMmWeightPtr);
        OPS_ERR_IF(CheckOptionalDims(context, mmXShape, mmWeightShape) != ge::GRAPH_SUCCESS,
                   VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), "CheckOptionalDims failed."),
                   return ge::GRAPH_FAILED);
        if (InferMMOutputShape(context, mmXShape, mmWeightShape, transMmWeightPtr, mmYShape) != ge::GRAPH_SUCCESS) {
            return ge::GRAPH_FAILED;
        }
    }

    if (InferPermuteOutputShape(context, permuteOutFlagPtr, e, a, h, permuteOutShape) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeAlltoAllvGroupedMatMul(gert::InferDataTypeContext *context)
{
    auto dType = context->GetInputDataType(INDEX_IN_GMM_X);
    auto *attrs = context->GetAttrs();
    const int64_t *yDtypePtr = nullptr;
    if (attrs->GetAttrNum() > INDEX_ATTR_Y_DTYPE_INDEX) {
        yDtypePtr = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_Y_DTYPE_INDEX);
    }
    const int64_t *mmDtypePtr = nullptr;
    if (attrs->GetAttrNum() > INDEX_ATTR_MM_DTYPE_INDEX) {
        mmDtypePtr = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_MM_DTYPE_INDEX);
    }
    ge::DataType yDataType =
        (yDtypePtr == nullptr || *yDtypePtr == UNDEFINED_VALUE) ? dType : static_cast<ge::DataType>(*yDtypePtr);
    ge::DataType mmDataType =
        (mmDtypePtr == nullptr || *mmDtypePtr == UNDEFINED_VALUE) ? dType : static_cast<ge::DataType>(*mmDtypePtr);
    context->SetOutputDataType(INDEX_OUT_MM_Y, mmDataType);
    OP_LOGD(context->GetNodeName(), "infershape mmY data type: %s.",
            TypeUtils::DataTypeToAscendString(mmDataType).GetString());
    context->SetOutputDataType(INDEX_OUT_GMM_Y, yDataType);
    OP_LOGD(context->GetNodeName(), "infershape gmmY data type: %s.",
            TypeUtils::DataTypeToAscendString(yDataType).GetString());
    context->SetOutputDataType(INDEX_PERMUTE_OUT, dType);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(AlltoAllvQuantGroupedMatMul)
    .InferShape(InferShapeAlltoAllvGroupedMatMul)
    .InferDataType(InferDataTypeAlltoAllvGroupedMatMul);
} // namespace ops
