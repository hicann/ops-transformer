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
 * \file grouped_mat_mul_allto_allv_infer_shape.cc
 * \brief
 */

#include "mc2_log.h"
#include "register/op_impl_registry.h"
#include "common/utils/op_mc2.h"
#include "mc2_moe_utils.h"
using namespace ge;
namespace ops {
static const size_t INDEX_IN_GMM_X = 0;
static const size_t INDEX_IN_GMM_WEIGHT = 1;
static const size_t INDEX_IN_MM_X = 4;
static const size_t INDEX_IN_MM_WEIGHT = 5;
static const size_t INDEX_OUT_GMM_Y = 0;
static const size_t INDEX_OUT_MM_Y = 1;
static const size_t INDEX_ATTR_EP_WORLD_SIZE = 1;
static const size_t INDEX_ATTR_SEND_COUNTS = 2;
static const size_t INDEX_ATTR_RECV_COUNTS = 3;
static const size_t INDEX_ATTR_TRANS_GMM_WEIGHT_INDEX = 4;
static const size_t INDEX_ATTR_TRANS_MM_WEIGHT_INDEX = 5;

static constexpr size_t DIM_NUM_0 = 0;
static constexpr size_t DIM_NUM_1 = 1;
static constexpr size_t DIM_NUM_2 = 2;
static constexpr size_t DIM_NUM_3 = 3;
static constexpr size_t DIM_0 = 0;
static constexpr size_t DIM_1 = 1;
static constexpr size_t DIM_2 = 2;

static constexpr int64_t FIRST_ELE_SIZE = -1;

static graphStatus CheckDims(const gert::InferShapeContext *context, const gert::Shape *gmmXShape,
                             const gert::Shape *gmmWeightShape, bool transGmmWeight)
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

    auto k1 = gmmXShape->GetDim(DIM_1);
    auto k2 = gmmWeightShape->GetDim(DIM_1);
    if (transGmmWeight) {
        k2 = gmmWeightShape->GetDim(DIM_2);
    }
    if (k1 != k2) {
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(),
                                            "Dim of gmmX and dim of gmmWeight do not match for MatMul");
        result = ge::GRAPH_FAILED;
    }
    return result;
}

static graphStatus CheckDimsOptional(const gert::InferShapeContext *context, const gert::Shape *mmXShape,
                                     const gert::Shape *mmWeightShape, bool transMmWeight)
{
    auto result = ge::GRAPH_SUCCESS;

    auto xDimNum = mmXShape->GetDimNum();
    auto mmWeightDimNum = mmWeightShape->GetDimNum();
    if (xDimNum != DIM_NUM_2) {
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), "Only x with dim 2 is supported.");
        result = ge::GRAPH_FAILED;
    }
    if (mmWeightDimNum != DIM_NUM_2) {
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), "Only mmWeight with dim 2 is supported.");
        result = ge::GRAPH_FAILED;
    }

    auto k1 = mmXShape->GetDim(DIM_1);
    auto k2 = mmWeightShape->GetDim(DIM_0);
    if (transMmWeight) {
        k2 = mmWeightShape->GetDim(DIM_1);
    }
    if (k1 != k2) {
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(),
                                            "Dim of x and dim of mmWeight do not match for MatMul");
        result = ge::GRAPH_FAILED;
    }
    return result;
}

static ge::graphStatus InferGMMOutputShape(const gert::InferShapeContext *context, gert::Shape *gmmYShape,
                                           const int64_t *epWorldSizePtr, const gert::ContinuousVector *recvCountsPtr,
                                           const gert::ContinuousVector *sendCountsPtr, const int64_t e, int64_t &bsk,
                                           const int64_t n1)
{
    gmmYShape->SetDim(DIM_0, FIRST_ELE_SIZE);
    gmmYShape->SetDim(DIM_1, FIRST_ELE_SIZE);
    gmmYShape->SetDimNum(DIM_NUM_2);
    if (e != FIRST_ELE_SIZE) {
        int64_t arraySize = e * (*epWorldSizePtr);
        OPS_ERR_IF(sendCountsPtr->GetSize() < static_cast<size_t>(arraySize),
                   VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(),
                                                       "sendCounts size should not be smaller than e * epWorldSize."),
                   return ge::GRAPH_FAILED);
        OPS_ERR_IF(recvCountsPtr->GetSize() < static_cast<size_t>(arraySize),
                   VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(),
                                                       "recvCounts size should not be smaller than e * epWorldSize."),
                   return ge::GRAPH_FAILED);
        OPS_ERR_IF(recvCountsPtr->GetData() == nullptr,
                   VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), "recvCounts GetData is nullptr."),
                   return ge::GRAPH_FAILED);
        for (int64_t i = 0UL; i < arraySize; i++) {
            bsk += static_cast<const int64_t *>(recvCountsPtr->GetData())[i];
        }
        gmmYShape->SetDim(DIM_1, n1);
        gmmYShape->SetDim(DIM_0, bsk);
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
    if ((mmWeightShape != nullptr) && (mmXShape != nullptr) && (mmYShape != nullptr) && (transMmWeightPtr != nullptr)) {
        OPS_ERR_IF(CheckDimsOptional(context, mmXShape, mmWeightShape, *transMmWeightPtr) != ge::GRAPH_SUCCESS,
                   VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), "CheckDimsOptional failed."),
                   return ge::GRAPH_FAILED);
        int64_t bs = mmXShape->GetDim(DIM_0);
        mmYShape->SetDimNum(DIM_NUM_2);
        mmYShape->SetDim(DIM_1, FIRST_ELE_SIZE);
        mmYShape->SetDim(DIM_0, FIRST_ELE_SIZE);
        if (bs != FIRST_ELE_SIZE) {
            int64_t n2 = *transMmWeightPtr ? mmWeightShape->GetDim(DIM_0) : mmWeightShape->GetDim(DIM_1);
            mmYShape->SetDim(DIM_0, bs);
            mmYShape->SetDim(DIM_1, n2);
        }
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeGroupedMatMulAlltoAllv(gert::InferShapeContext *context)
{
    auto *gmmXShape = context->GetInputShape(INDEX_IN_GMM_X);
    auto *gmmWeightShape = context->GetInputShape(INDEX_IN_GMM_WEIGHT);
    auto *mmXShape = context->GetOptionalInputShape(INDEX_IN_MM_X);
    auto *mmWeightShape = context->GetOptionalInputShape(INDEX_IN_MM_WEIGHT);
    auto *gmmYShape = context->GetOutputShape(INDEX_OUT_GMM_Y);
    auto *mmYShape = context->GetOutputShape(INDEX_OUT_MM_Y);

    auto *attrs = context->GetAttrs();
    auto *epWorldSizePtr = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_EP_WORLD_SIZE);
    auto *recvCountsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(INDEX_ATTR_RECV_COUNTS);
    auto *sendCountsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(INDEX_ATTR_SEND_COUNTS);
    auto *transGmmWeightPtr = attrs->GetAttrPointer<bool>(INDEX_ATTR_TRANS_GMM_WEIGHT_INDEX);
    auto *transMmWeightPtr = attrs->GetAttrPointer<bool>(INDEX_ATTR_TRANS_MM_WEIGHT_INDEX);

    OPS_CHECK_NULL_WITH_CONTEXT(context, gmmXShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context, gmmWeightShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context, gmmYShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context, epWorldSizePtr);
    OPS_CHECK_NULL_WITH_CONTEXT(context, recvCountsPtr);
    OPS_CHECK_NULL_WITH_CONTEXT(context, sendCountsPtr);
    OPS_CHECK_NULL_WITH_CONTEXT(context, transGmmWeightPtr);
    OPS_ERR_IF(CheckDims(context, gmmXShape, gmmWeightShape, *transGmmWeightPtr) != ge::GRAPH_SUCCESS,
               VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), "CheckDims failed."),
               return ge::GRAPH_FAILED);

    int64_t e = gmmWeightShape->GetDim(DIM_0);
    int64_t bsk = 0;
    int64_t n1 = *transGmmWeightPtr ? gmmWeightShape->GetDim(DIM_1) : gmmWeightShape->GetDim(DIM_2);

    ge::graphStatus ret =
        InferGMMOutputShape(context, gmmYShape, epWorldSizePtr, recvCountsPtr, sendCountsPtr, e, bsk, n1);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = InferMMOutputShape(context, mmXShape, mmWeightShape, transMmWeightPtr, mmYShape);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeGroupedMatMulAlltoAllv(gert::InferDataTypeContext *context)
{
    auto dType = context->GetInputDataType(INDEX_IN_GMM_X);
    context->SetOutputDataType(INDEX_OUT_GMM_Y, dType);
    context->SetOutputDataType(INDEX_OUT_MM_Y, dType);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(GroupedMatMulAlltoAllv)
    .InferShape(InferShapeGroupedMatMulAlltoAllv)
    .InferDataType(InferDataTypeGroupedMatMulAlltoAllv);
} // namespace ops
