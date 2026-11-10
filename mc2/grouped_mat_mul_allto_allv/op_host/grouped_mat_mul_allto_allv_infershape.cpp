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

#include <limits>
#include <string>
#include <vector>
#include "mc2_log.h"
#include "register/op_impl_registry.h"
#include "common/utils/op_mc2.h"
#include "mc2_moe_utils.h"
using namespace ge;
namespace ops {

static void ReportGroupedInferMetadata(const char* opName, const std::string& reason, const char* code)
{
    // Requested metadata codes need a registration dependency outside this scope.
    (void)code;
    const char* name = opName == nullptr ? "GroupedMatMulAlltoAllv" : opName;
    OP_LOGE(name, "%s", reason.c_str());
}

static constexpr int64_t FIRST_ELE_SIZE = -1;

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

static constexpr size_t DIM_0 = 0;
static constexpr size_t DIM_1 = 1;
static constexpr size_t DIM_2 = 2;
static constexpr size_t DIM_NUM_0 = 0;
static constexpr size_t DIM_NUM_1 = 1;
static constexpr size_t DIM_NUM_2 = 2;
static constexpr size_t DIM_NUM_3 = 3;

static graphStatus CheckDims(const gert::InferShapeContext* context, const gert::Shape* gmmXShape,
                             const gert::Shape* gmmWeightShape, bool transGmmWeight)
{
    auto gmmXDimNum = gmmXShape->GetDimNum();
    auto gmmWeightDimNum = gmmWeightShape->GetDimNum();
    if (gmmXDimNum != DIM_NUM_2) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "gmm_x", std::to_string(gmmXDimNum).c_str(), "2");
        return ge::GRAPH_FAILED;
    }
    if (gmmWeightDimNum != DIM_NUM_3) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "gmm_weight", std::to_string(gmmWeightDimNum).c_str(),
                                     "3");
        return ge::GRAPH_FAILED;
    }

    auto k2 = gmmWeightShape->GetDim(DIM_1);
    auto k1 = gmmXShape->GetDim(DIM_1);
    if (transGmmWeight) {
        k2 = gmmWeightShape->GetDim(DIM_2);
    }
    if (k2 >= 0 && k1 >= 0 && k2 != k1) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "gmm_x, gmm_weight",
                                              (std::to_string(k1) + ", " + std::to_string(k2)).c_str(),
                                              "Contraction dimensions must match");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static graphStatus CheckDimsOptional(const gert::InferShapeContext* context, const gert::Shape* mmXShape,
                                     const gert::Shape* mmWeightShape, bool transMmWeight)
{
    auto xDimNum = mmXShape->GetDimNum();
    auto mmWeightDimNum = mmWeightShape->GetDimNum();
    if (xDimNum != DIM_NUM_2) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "mm_x", std::to_string(xDimNum).c_str(), "2");
        return ge::GRAPH_FAILED;
    }
    if (mmWeightDimNum != DIM_NUM_2) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(context->GetNodeName(), "mm_weight", std::to_string(mmWeightDimNum).c_str(), "2");
        return ge::GRAPH_FAILED;
    }

    auto k2 = mmWeightShape->GetDim(DIM_0);
    auto k1 = mmXShape->GetDim(DIM_1);
    if (transMmWeight) {
        k2 = mmWeightShape->GetDim(DIM_1);
    }
    if (k2 >= 0 && k1 >= 0 && k2 != k1) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(context->GetNodeName(), "mm_x, mm_weight",
                                              (std::to_string(k1) + ", " + std::to_string(k2)).c_str(),
                                              "Contraction dimensions must match");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferGMMOutputShape(const gert::InferShapeContext* context, gert::Shape* gmmYShape,
                                           const int64_t* epWorldSizePtr, const gert::ContinuousVector* recvCountsPtr,
                                           const gert::ContinuousVector* sendCountsPtr, const int64_t e, int64_t& bsk,
                                           const int64_t n1)
{
    gmmYShape->SetDimNum(DIM_NUM_2);
    gmmYShape->SetDim(DIM_0, FIRST_ELE_SIZE);
    gmmYShape->SetDim(DIM_1, FIRST_ELE_SIZE);
    if (e != FIRST_ELE_SIZE) {
        OPS_CHECK(e < 0 || e > std::numeric_limits<int64_t>::max() / *epWorldSizePtr,
                  OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                      context->GetNodeName(), "gmm_weight", std::to_string(e).c_str(),
                      "Expert count must be nonnegative and its product with ep_world_size must fit int64"),
                  return ge::GRAPH_FAILED);
        int64_t arraySize = e * (*epWorldSizePtr);
        OPS_ERR_IF(
            sendCountsPtr->GetSize() < static_cast<size_t>(arraySize),
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "send_counts",
                                                     ("element count=" + std::to_string(sendCountsPtr->GetSize()) +
                                                      ", required at least=" + std::to_string(arraySize))
                                                         .c_str()),
            return ge::GRAPH_FAILED);
        OPS_ERR_IF(
            recvCountsPtr->GetSize() < static_cast<size_t>(arraySize),
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "recv_counts",
                                                     ("element count=" + std::to_string(recvCountsPtr->GetSize()) +
                                                      ", required at least=" + std::to_string(arraySize))
                                                         .c_str()),
            return ge::GRAPH_FAILED);
        OPS_ERR_IF(recvCountsPtr->GetData() == nullptr,
                   OP_LOGE_WITHOUT_REPORT(context->GetNodeName(),
                                          "ContinuousVector inline recv_counts storage is unexpectedly null"),
                   return ge::GRAPH_FAILED);
        for (int64_t i = 0UL; i < arraySize; i++) {
            const int64_t count = static_cast<const int64_t*>(recvCountsPtr->GetData())[i];
            OPS_CHECK(count < 0 || count > std::numeric_limits<int64_t>::max() - bsk,
                      OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                          context->GetNodeName(), "recv_counts",
                          ("index=" + std::to_string(i) + ", actual=" + std::to_string(count) +
                           ", prior sum=" + std::to_string(bsk) + "; expected nonnegative counts with int64 sum")
                              .c_str()),
                      return ge::GRAPH_FAILED);
            bsk += count;
        }
        gmmYShape->SetDim(DIM_0, bsk);
        gmmYShape->SetDim(DIM_1, n1);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferMMOutputShape(const gert::InferShapeContext* context, const gert::Shape* mmXShape,
                                          const gert::Shape* mmWeightShape, const bool* transMmWeightPtr,
                                          gert::Shape* mmYShape)
{
    if (mmXShape == nullptr && mmWeightShape == nullptr && mmYShape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    OPS_ERR_IF(mmYShape == nullptr,
               ReportGroupedInferMetadata(context->GetNodeName(), "Output shape holder mm_y is absent", "EZ1014"),
               return ge::GRAPH_FAILED);
    mmYShape->SetDimNum(DIM_NUM_0);
    if ((mmWeightShape != nullptr) && (mmXShape != nullptr) && (mmYShape != nullptr) && (transMmWeightPtr != nullptr)) {
        OPS_ERR_IF(CheckDimsOptional(context, mmXShape, mmWeightShape, *transMmWeightPtr) != ge::GRAPH_SUCCESS,
                   OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "CheckDimsOptional failed."),
                   return ge::GRAPH_FAILED);
        int64_t bs = mmXShape->GetDim(DIM_0);
        mmYShape->SetDimNum(DIM_NUM_2);
        mmYShape->SetDim(DIM_0, FIRST_ELE_SIZE);
        mmYShape->SetDim(DIM_1, FIRST_ELE_SIZE);
        if (bs != FIRST_ELE_SIZE) {
            int64_t n2 = *transMmWeightPtr ? mmWeightShape->GetDim(DIM_0) : mmWeightShape->GetDim(DIM_1);
            mmYShape->SetDim(DIM_1, n2);
            mmYShape->SetDim(DIM_0, bs);
        }
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeGroupedMatMulAlltoAllv(gert::InferShapeContext* context)
{
    if (context == nullptr) {
        ReportGroupedInferMetadata("grouped_mat_mul_allto_allv", "Inference context is null", "EZ1014");
        return GRAPH_FAILED;
    }
    auto* gmmXShape = context->GetInputShape(INDEX_IN_GMM_X);
    auto* gmmWeightShape = context->GetInputShape(INDEX_IN_GMM_WEIGHT);
    auto* mmXShape = context->GetOptionalInputShape(INDEX_IN_MM_X);
    auto* mmWeightShape = context->GetOptionalInputShape(INDEX_IN_MM_WEIGHT);
    auto* gmmYShape = context->GetOutputShape(INDEX_OUT_GMM_Y);
    auto* mmYShape = context->GetOutputShape(INDEX_OUT_MM_Y);

    OPS_CHECK(gmmXShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "gmm_x"), return GRAPH_FAILED);
    OPS_CHECK(gmmWeightShape == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "gmm_weight"),
              return GRAPH_FAILED);
    OPS_CHECK(gmmYShape == nullptr,
              ReportGroupedInferMetadata(context->GetNodeName(), "Output shape holder y is absent", "EZ1014"),
              return GRAPH_FAILED);

    auto* attrs = context->GetAttrs();
    OPS_CHECK(attrs == nullptr,
              ReportGroupedInferMetadata(context->GetNodeName(), "RuntimeAttrs holder is absent", "EZ1014"),
              return GRAPH_FAILED);
    auto* epWorldSizePtr = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_EP_WORLD_SIZE);
    auto* recvCountsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(INDEX_ATTR_RECV_COUNTS);
    auto* sendCountsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(INDEX_ATTR_SEND_COUNTS);
    auto* transGmmWeightPtr = attrs->GetAttrPointer<bool>(INDEX_ATTR_TRANS_GMM_WEIGHT_INDEX);
    auto* transMmWeightPtr = attrs->GetAttrPointer<bool>(INDEX_ATTR_TRANS_MM_WEIGHT_INDEX);

    OPS_CHECK(epWorldSizePtr == nullptr,
              OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "ep_world_size",
                                                       "Required attribute is absent"),
              return GRAPH_FAILED);
    OPS_CHECK(
        recvCountsPtr == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "recv_counts", "Required attribute is absent"),
        return GRAPH_FAILED);
    OPS_CHECK(
        sendCountsPtr == nullptr,
        OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(context->GetNodeName(), "send_counts", "Required attribute is absent"),
        return GRAPH_FAILED);
    OPS_CHECK(transGmmWeightPtr == nullptr,
              ReportGroupedInferMetadata(context->GetNodeName(), "Default trans_gmm_weight is absent from RuntimeAttrs",
                                         "EZ1014"),
              return GRAPH_FAILED);
    OPS_CHECK(*epWorldSizePtr <= 0,
              OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON(
                  context->GetNodeName(), "ep_world_size",
                  ("actual=" + std::to_string(*epWorldSizePtr) + "; expected positive world size").c_str()),
              return GRAPH_FAILED);
    OPS_CHECK(mmXShape != nullptr && mmWeightShape != nullptr && transMmWeightPtr == nullptr,
              ReportGroupedInferMetadata(context->GetNodeName(), "Default trans_mm_weight is absent from RuntimeAttrs",
                                         "EZ1014"),
              return GRAPH_FAILED);
    OPS_ERR_IF(CheckDims(context, gmmXShape, gmmWeightShape, *transGmmWeightPtr) != ge::GRAPH_SUCCESS,
               OP_LOGE_WITHOUT_REPORT(context->GetNodeName(), "CheckDims failed."), return ge::GRAPH_FAILED);

    int64_t bsk = 0;
    int64_t e = gmmWeightShape->GetDim(DIM_0);
    int64_t n1 = *transGmmWeightPtr ? gmmWeightShape->GetDim(DIM_1) : gmmWeightShape->GetDim(DIM_2);

    ge::graphStatus ret = InferMMOutputShape(context, mmXShape, mmWeightShape, transMmWeightPtr, mmYShape);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }

    ret = InferGMMOutputShape(context, gmmYShape, epWorldSizePtr, recvCountsPtr, sendCountsPtr, e, bsk, n1);
    if (ret != ge::GRAPH_SUCCESS) {
        return ret;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeGroupedMatMulAlltoAllv(gert::InferDataTypeContext* context)
{
    if (context == nullptr) {
        ReportGroupedInferMetadata("grouped_mat_mul_allto_allv", "Inference context is null", "EZ1009");
        return GRAPH_FAILED;
    }
    auto dType = context->GetInputDataType(INDEX_IN_GMM_X);
    OPS_CHECK(dType == DT_UNDEFINED,
              ReportGroupedInferMetadata(context->GetNodeName(), "gmm_x dtype metadata is absent", "EZ1009"),
              return GRAPH_FAILED);
    OPS_CHECK(context->SetOutputDataType(INDEX_OUT_GMM_Y, dType) != GRAPH_SUCCESS,
              ReportGroupedInferMetadata(context->GetNodeName(), "Output dtype slot y is absent", "EZ1009"),
              return GRAPH_FAILED);
    const bool hasSharedMm = context->GetOptionalInputDataType(INDEX_IN_MM_X) != DT_UNDEFINED &&
                             context->GetOptionalInputDataType(INDEX_IN_MM_WEIGHT) != DT_UNDEFINED;
    if (hasSharedMm || context->GetComputeNodeOutputNum() > INDEX_OUT_MM_Y) {
        OPS_CHECK(context->SetOutputDataType(INDEX_OUT_MM_Y, dType) != GRAPH_SUCCESS,
                  ReportGroupedInferMetadata(context->GetNodeName(), "Output dtype slot mm_y is absent", "EZ1009"),
                  return GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(GroupedMatMulAlltoAllv)
    .InferShape(InferShapeGroupedMatMulAlltoAllv)
    .InferDataType(InferDataTypeGroupedMatMulAlltoAllv);
} // namespace ops
