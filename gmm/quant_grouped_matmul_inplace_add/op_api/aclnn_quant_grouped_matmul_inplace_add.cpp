/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_quant_grouped_matmul_inplace_add.h"

#include <dlfcn.h>
#include <new>
#include <sstream>

#include "aclnn_kernels/transdata.h"
#include "aclnn_kernels/contiguous.h"
#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"
#include "log/log.h"

#include "../../grouped_matmul/op_api/grouped_matmul_util.h"
#include "../../grouped_matmul/op_api/grouped_matmul_950_checker.h"
#include "quant_grouped_matmul_inplace_add_util.h"
#include "quant_grouped_matmul_inplace_add.h"
#include "quant_grouped_matmul_inplace_add_950_checker.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {
constexpr const char *QGMM_INPLACE_ADD_ACLNN_OP_NAME = "aclnnQuantGroupedMatmulInplaceAddGetWorkspaceSize";

std::string ShapeToStringWithoutBracket(const op::Shape &shape)
{
    std::ostringstream oss;
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        if (i != 0) {
            oss << ", ";
        }
        oss << shape.GetDim(i);
    }
    return oss.str();
}

std::string ViewShapeToString(const aclTensor *tensor)
{
    return ShapeToStringWithoutBracket(tensor->GetViewShape());
}

#define QGMM_INPLACE_ADD_CHECK_REPORT(cond, retExpr, reportExpr) \
    do { \
        if (!(cond)) { \
            reportExpr; \
            retExpr; \
        } \
    } while (0)

static aclnnStatus CheckX1X2NotNull(QGmmInPlaceAdd::QuantGroupedMatmulInplaceAddParams params)
{
    QGMM_INPLACE_ADD_CHECK_REPORT(params.x1 != nullptr, return ACLNN_ERR_PARAM_NULLPTR,
                                  OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(QGMM_INPLACE_ADD_ACLNN_OP_NAME, "x1", "nullptr",
                                                                        "the value of x1 cannot be nullptr"));
    QGMM_INPLACE_ADD_CHECK_REPORT(params.x2 != nullptr, return ACLNN_ERR_PARAM_NULLPTR,
                                  OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(QGMM_INPLACE_ADD_ACLNN_OP_NAME, "x2", "nullptr",
                                                                        "the value of x2 cannot be nullptr"));
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckNotNull(QGmmInPlaceAdd::QuantGroupedMatmulInplaceAddParams params)
{
    CHECK_RET(CheckX1X2NotNull(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR);
    QGMM_INPLACE_ADD_CHECK_REPORT(
        params.scale2 != nullptr, return ACLNN_ERR_PARAM_NULLPTR,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(QGMM_INPLACE_ADD_ACLNN_OP_NAME, "scale2", "nullptr",
                                              "the value of scale2 cannot be nullptr"));
    QGMM_INPLACE_ADD_CHECK_REPORT(
        params.groupList != nullptr, return ACLNN_ERR_PARAM_NULLPTR,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(QGMM_INPLACE_ADD_ACLNN_OP_NAME, "groupList", "nullptr",
                                              "the value of groupList cannot be nullptr"));
    QGMM_INPLACE_ADD_CHECK_REPORT(
        params.yRef != nullptr, return ACLNN_ERR_PARAM_NULLPTR,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(QGMM_INPLACE_ADD_ACLNN_OP_NAME, "yRef", "nullptr",
                                              "the value of yRef cannot be nullptr"));
    QGMM_INPLACE_ADD_CHECK_REPORT(
        params.scale1Optional != nullptr, return ACLNN_ERR_PARAM_NULLPTR,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(QGMM_INPLACE_ADD_ACLNN_OP_NAME, "scale1", "nullptr",
                                              "the value of scale1 cannot be nullptr"));
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckFormat(QGmmInPlaceAdd::QuantGroupedMatmulInplaceAddParams params)
{
    QGMM_INPLACE_ADD_CHECK_REPORT(
        params.x1->GetStorageFormat() == Format::FORMAT_ND, return ACLNN_ERR_PARAM_INVALID,
        OP_LOGE_FOR_INVALID_FORMAT(QGMM_INPLACE_ADD_ACLNN_OP_NAME, "x1",
                                   op::ToString(params.x1->GetStorageFormat()).GetString(), "ND"));
    QGMM_INPLACE_ADD_CHECK_REPORT(
        params.x2->GetStorageFormat() == Format::FORMAT_ND, return ACLNN_ERR_PARAM_INVALID,
        OP_LOGE_FOR_INVALID_FORMAT(QGMM_INPLACE_ADD_ACLNN_OP_NAME, "x2",
                                   op::ToString(params.x2->GetStorageFormat()).GetString(), "ND"));
    if (!(params.scale2->GetStorageFormat() == Format::FORMAT_ND ||
          params.scale2->GetStorageFormat() == Format::FORMAT_NCL)) {
        OP_LOGE_FOR_INVALID_FORMAT(QGMM_INPLACE_ADD_ACLNN_OP_NAME, "scale2",
                                   op::ToString(params.scale2->GetStorageFormat()).GetString(), "ND or NCL");
        return ACLNN_ERR_PARAM_INVALID;
    }
    QGMM_INPLACE_ADD_CHECK_REPORT(
        params.groupList->GetStorageFormat() == Format::FORMAT_ND, return ACLNN_ERR_PARAM_INVALID,
        OP_LOGE_FOR_INVALID_FORMAT(QGMM_INPLACE_ADD_ACLNN_OP_NAME, "groupList",
                                   op::ToString(params.groupList->GetStorageFormat()).GetString(), "ND"));
    if (!(params.yRef->GetStorageFormat() == Format::FORMAT_ND ||
          params.yRef->GetStorageFormat() == Format::FORMAT_NCL)) {
        OP_LOGE_FOR_INVALID_FORMAT(QGMM_INPLACE_ADD_ACLNN_OP_NAME, "yRef",
                                   op::ToString(params.yRef->GetStorageFormat()).GetString(), "ND or NCL");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (!(params.scale1Optional->GetStorageFormat() == Format::FORMAT_ND ||
          params.scale1Optional->GetStorageFormat() == Format::FORMAT_NCL)) {
        OP_LOGE_FOR_INVALID_FORMAT(QGMM_INPLACE_ADD_ACLNN_OP_NAME, "scale1",
                                   op::ToString(params.scale1Optional->GetStorageFormat()).GetString(), "ND or NCL");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus IsMxQuantDim(QGmmInPlaceAdd::QuantGroupedMatmulInplaceAddParams params)
{
    auto x1ScaleDimNum = params.scale1Optional->GetViewShape().GetDimNum();
    auto x2ScaleDimNum = params.scale2->GetViewShape().GetDimNum();
    QGMM_INPLACE_ADD_CHECK_REPORT(
        x2ScaleDimNum == gmm::MX_SPLIT_K_SCALE_DIM, return ACLNN_ERR_PARAM_INVALID,
        OP_LOGE_FOR_INVALID_SHAPEDIM(QGMM_INPLACE_ADD_ACLNN_OP_NAME, "scale2", std::to_string(x2ScaleDimNum),
                                     std::to_string(gmm::MX_SPLIT_K_SCALE_DIM)));
    QGMM_INPLACE_ADD_CHECK_REPORT(
        x1ScaleDimNum == gmm::MX_SPLIT_K_PER_TOKEN_SCALE_DIM, return ACLNN_ERR_PARAM_INVALID,
        OP_LOGE_FOR_INVALID_SHAPEDIM(QGMM_INPLACE_ADD_ACLNN_OP_NAME, "scale1", std::to_string(x1ScaleDimNum),
                                     std::to_string(gmm::MX_SPLIT_K_PER_TOKEN_SCALE_DIM)));
    auto scale1LastDimValue = params.scale1Optional->GetViewShape().GetDim(gmm::MX_SPLIT_K_PER_TOKEN_SCALE_DIM - 1);
    auto scale2LastDimValue = params.scale2->GetViewShape().GetDim(gmm::MX_SPLIT_K_SCALE_DIM - 1);
    QGMM_INPLACE_ADD_CHECK_REPORT(
        scale1LastDimValue == 2, return ACLNN_ERR_PARAM_INVALID,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(QGMM_INPLACE_ADD_ACLNN_OP_NAME, "scale1",
                                              ViewShapeToString(params.scale1Optional),
                                              "in mx quant mode, last dim of scale1 must be equal to 2"));
    QGMM_INPLACE_ADD_CHECK_REPORT(scale2LastDimValue == 2, return ACLNN_ERR_PARAM_INVALID,
                                  OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                                      QGMM_INPLACE_ADD_ACLNN_OP_NAME, "scale2", ViewShapeToString(params.scale2),
                                      "in mx quant mode, last dim of scale2 must be equal to 2"));
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckShape(QGmmInPlaceAdd::QuantGroupedMatmulInplaceAddParams params)
{
    auto x2DimNum = params.x2->GetViewShape().GetDimNum();
    auto x1DimNum = params.x1->GetViewShape().GetDimNum();
    auto groupListDimNum = params.groupList->GetViewShape().GetDimNum();
    auto yDimNum = params.yRef->GetViewShape().GetDimNum();
    QGMM_INPLACE_ADD_CHECK_REPORT(
        x1DimNum == 2, return ACLNN_ERR_PARAM_INVALID,
        OP_LOGE_FOR_INVALID_SHAPEDIM(QGMM_INPLACE_ADD_ACLNN_OP_NAME, "x1", std::to_string(x1DimNum), "2"));
    QGMM_INPLACE_ADD_CHECK_REPORT(
        x2DimNum == 2, return ACLNN_ERR_PARAM_INVALID,
        OP_LOGE_FOR_INVALID_SHAPEDIM(QGMM_INPLACE_ADD_ACLNN_OP_NAME, "x2", std::to_string(x2DimNum), "2"));
    QGMM_INPLACE_ADD_CHECK_REPORT(groupListDimNum == 1, return ACLNN_ERR_PARAM_INVALID,
                                  OP_LOGE_FOR_INVALID_SHAPEDIM(QGMM_INPLACE_ADD_ACLNN_OP_NAME, "groupList",
                                                               std::to_string(groupListDimNum), "1"));
    QGMM_INPLACE_ADD_CHECK_REPORT(
        yDimNum == 3, return ACLNN_ERR_PARAM_INVALID,
        OP_LOGE_FOR_INVALID_SHAPEDIM(QGMM_INPLACE_ADD_ACLNN_OP_NAME, "yRef", std::to_string(yDimNum), "3"));
    auto aKDim = params.x1->GetViewShape().GetDim(1);
    auto bKDim = params.x2->GetViewShape().GetDim(0);
    auto nDim = params.x2->GetViewShape().GetDim(1);
    auto mDim = params.x1->GetViewShape().GetDim(0);
    auto gDim = params.groupList->GetViewShape().GetDim(0);

    auto yGDim = params.yRef->GetViewShape().GetDim(0);
    auto yMDim = params.yRef->GetViewShape().GetDim(1);
    auto yNDim = params.yRef->GetViewShape().GetDim(2);

    QGMM_INPLACE_ADD_CHECK_REPORT(
        mDim >= 0, return ACLNN_ERR_PARAM_INVALID,
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(QGMM_INPLACE_ADD_ACLNN_OP_NAME, "x1", ViewShapeToString(params.x1),
                                              "axis M of x1 must be a positive number"));

    QGMM_INPLACE_ADD_CHECK_REPORT(aKDim == bKDim, return ACLNN_ERR_PARAM_INVALID,
                                  OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                                      QGMM_INPLACE_ADD_ACLNN_OP_NAME, "x1 and x2",
                                      "x1=" + ViewShapeToString(params.x1) + ", x2=" + ViewShapeToString(params.x2),
                                      "axis K of x1 must be equal to axis K of x2"));
    QGMM_INPLACE_ADD_CHECK_REPORT(
        gDim == yGDim && mDim == yMDim && nDim == yNDim, return ACLNN_ERR_PARAM_INVALID,
        OP_LOGE_FOR_INVALID_SHAPE(
            QGMM_INPLACE_ADD_ACLNN_OP_NAME, "yRef",
            "(" + std::to_string(yGDim) + ", " + std::to_string(yMDim) + ", " + std::to_string(yNDim) + ")",
            "(" + std::to_string(gDim) + ", " + std::to_string(mDim) + ", " + std::to_string(nDim) + ")"));
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckDtype(QGmmInPlaceAdd::QuantGroupedMatmulInplaceAddParams params)
{
    auto x1Dtype = params.x1->GetDataType();
    auto x2Dtype = params.x2->GetDataType();
    QGMM_INPLACE_ADD_CHECK_REPORT(
        params.yRef->GetDataType() == DataType::DT_FLOAT, return ACLNN_ERR_PARAM_INVALID,
        OP_LOGE_FOR_INVALID_DTYPE(QGMM_INPLACE_ADD_ACLNN_OP_NAME, "yRef",
                                  op::ToString(params.yRef->GetDataType()).GetString(), "FLOAT32"));
    QGMM_INPLACE_ADD_CHECK_REPORT(
        params.groupList->GetDataType() == DataType::DT_INT64, return ACLNN_ERR_PARAM_INVALID,
        OP_LOGE_FOR_INVALID_DTYPE(QGMM_INPLACE_ADD_ACLNN_OP_NAME, "groupList",
                                  op::ToString(params.groupList->GetDataType()).GetString(), "INT64"));
    if ((x1Dtype == DataType::DT_FLOAT8_E4M3FN || x1Dtype == DataType::DT_FLOAT8_E5M2) &&
        (x2Dtype == DataType::DT_FLOAT8_E4M3FN || x2Dtype == DataType::DT_FLOAT8_E5M2)) {
        CHECK_COND(IsMxQuantDim(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID, "Check IsMxQuantDim failed.");
        QGMM_INPLACE_ADD_CHECK_REPORT(
            params.scale2->GetDataType() == DataType::DT_FLOAT8_E8M0, return ACLNN_ERR_PARAM_INVALID,
            OP_LOGE_FOR_INVALID_DTYPE(QGMM_INPLACE_ADD_ACLNN_OP_NAME, "scale2",
                                      op::ToString(params.scale2->GetDataType()).GetString(), "FLOAT8_E8M0"));
        QGMM_INPLACE_ADD_CHECK_REPORT(
            params.scale1Optional->GetDataType() == DataType::DT_FLOAT8_E8M0, return ACLNN_ERR_PARAM_INVALID,
            OP_LOGE_FOR_INVALID_DTYPE(QGMM_INPLACE_ADD_ACLNN_OP_NAME, "scale1",
                                      op::ToString(params.scale1Optional->GetDataType()).GetString(), "FLOAT8_E8M0"));
    } else if (!(x1Dtype == DataType::DT_HIFLOAT8 && x2Dtype == DataType::DT_HIFLOAT8)) {
        OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
            QGMM_INPLACE_ADD_ACLNN_OP_NAME, "x1 and x2",
            "x1=" + std::string(op::ToString(x1Dtype).GetString()) + ", x2=" + op::ToString(x2Dtype).GetString(),
            "in quant case, the dtypes of x1 and x2 must be within the range both FLOAT8_E4M3FN/FLOAT8_E5M2 "
            "or both HIFLOAT8");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckParams(QGmmInPlaceAdd::QuantGroupedMatmulInplaceAddParams params)
{
    if (!(params.groupListType == 0 || params.groupListType == 1)) {
        OP_LOGE_FOR_INVALID_VALUE(QGMM_INPLACE_ADD_ACLNN_OP_NAME, "groupListType", std::to_string(params.groupListType),
                                  "0 or 1");
        return ACLNN_ERR_PARAM_INVALID;
    }
    QGMM_INPLACE_ADD_CHECK_REPORT(
        params.groupSize == 0, return ACLNN_ERR_PARAM_INVALID,
        OP_LOGE_FOR_INVALID_VALUE(QGMM_INPLACE_ADD_ACLNN_OP_NAME, "groupSize", std::to_string(params.groupSize), "0"));
    CHECK_RET(CheckNotNull(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckFormat(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckDtype(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    gmm::GroupedMatmulParamsBase<aclTensor> gmmParams;
    gmmParams.x = params.x1;
    gmmParams.weight = params.x2;
    gmmParams.scaleOptional = params.scale2;
    gmmParams.perTokenScaleOptional = params.scale1Optional;
    gmmParams.y = params.yRef;
    gmmParams.groupTensorOptional = params.groupList;
    gmmParams.groupListType = params.groupListType;
    gmmParams.groupType = gmm::SPLIT_K;
    gmmParams.xDtype = params.x1->GetDataType();
    gmmParams.transposeX = true;
    gmmParams.transposeWeight = false;
    if (params.x1->GetDataType() == DataType::DT_HIFLOAT8 && params.x2->GetDataType() == DataType::DT_HIFLOAT8) {
        auto checker = QGmmInPlaceAdd::AclnnQuantGroupedMatmulInplaceAddDAV3510Checker<aclTensor>(gmmParams);
        checker.SetInputName("x1", "x2", "scale1Optional", "scale2", "groupList");
        CHECK_RET(checker.CheckQuantGroupedMatmulInplaceAddDAV3510() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    } else {
        auto checker = gmm::AclnnGroupedMatmulDAV3510Checker<aclTensor>(gmmParams);
        checker.SetInputName("x1", "x2", "scale1Optional", "scale2", "groupList");
        CHECK_RET(checker.CheckGroupedMatmulDAV3510() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus SetTransViewShape(const aclTensor *&inputTensor, aclOpExecutor *executor)
{
    op::Shape viewShape = inputTensor->GetViewShape();
    uint32_t viewShapeDimsNum = viewShape.GetDimNum();
    CHECK_RET(viewShapeDimsNum >= 2, ACLNN_ERR_PARAM_INVALID);
    op::Shape shape;
    shape.SetScalar();
    // 2: the second last dimension; in for-loops, it indicates dimensions before the second last remain unchanged.
    for (uint32_t i = 0; i < viewShapeDimsNum - 2; ++i) {
        shape.AppendDim(viewShape.GetDim(i));
    }
    // viewShapeDimsNum - 1, the dim value of the last dim. viewShapeDimsNum - 2, the dim value of the second last dim.
    shape.AppendDim(viewShape.GetDim(viewShapeDimsNum - 1));
    shape.AppendDim(viewShape.GetDim(viewShapeDimsNum - 2)); // 2:the second last dim.
    inputTensor = executor->CreateView(inputTensor, shape, inputTensor->GetViewOffset());
    CHECK_RET(inputTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus SetTransViewShapeForPertoken(const aclTensor *&inputTensor, aclOpExecutor *executor)
{
    op::Shape viewShape = inputTensor->GetViewShape();
    op::Shape shape;
    shape.SetScalar();
    if (viewShape.GetDimNum() < 3) { // only pertoken in mx typek quant mode have to trans, which dim num is 3
        OP_LOGE_FOR_INVALID_SHAPEDIM(QGMM_INPLACE_ADD_ACLNN_OP_NAME, "scale1", std::to_string(viewShape.GetDimNum()),
                                     "3");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // swap first two dim
    shape.AppendDim(viewShape.GetDim(1));
    shape.AppendDim(viewShape.GetDim(0));
    shape.AppendDim(viewShape.GetDim(2)); // 2 is last dim contiguous in k axis in mx typek quant mode
    inputTensor =
        executor->CreateView(inputTensor, shape, inputTensor->GetViewOffset()); // use executor to create tensor
    CHECK_RET(inputTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus DataContiguous(const aclTensor *&tensor, aclOpExecutor *executor)
{
    tensor = l0op::Contiguous(tensor, executor);
    CHECK_RET(tensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus ParamsDataContiguous(QGmmInPlaceAdd::QuantGroupedMatmulInplaceAddParams &params,
                                        aclOpExecutor *executorPtr)
{
    CHECK_COND(DataContiguous(params.x1, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Contiguous x1 failed.");
    CHECK_COND(DataContiguous(params.x2, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Contiguous x2 failed.");
    CHECK_COND(DataContiguous(params.scale1Optional, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Contiguous scale1Optional failed.");
    CHECK_COND(DataContiguous(params.scale2, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Contiguous scale2 failed.");
    CHECK_COND(DataContiguous(params.groupList, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Contiguous groupList failed.");
    return ACLNN_SUCCESS;
}

static bool IsSpecialTranspose(const aclTensor *const inputTensor)
{
    const auto inputShape = inputTensor->GetViewShape();
    int64_t dim1 = inputShape.GetDimNum() - gmm::LAST_FIRST_DIM_INDEX;
    int64_t dim2 = inputShape.GetDimNum() - gmm::LAST_SECOND_DIM_INDEX;
    return inputShape.GetDim(dim1) == 1 && inputShape.GetDim(dim2) == 1;
}

static aclnnStatus aclnnQuantGroupedMatmulInplaceAddGetWorkspaceSizeCommon(
    QGmmInPlaceAdd::QuantGroupedMatmulInplaceAddParams params, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    // 固定写法，创建OpExecutor
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    auto ret = CheckX1X2NotNull(params);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto executorPtr = uniqueExecutor.get();
    auto x1MDim = params.x1->GetViewShape().GetDim(0);
    auto x2NDim = params.x2->GetViewShape().GetDim(1);
    if (x1MDim == 0 || x2NDim == 0) {
        *workspaceSize = 0UL;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }
    // 固定写法，参数检查
    ret = CheckParams(params);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    bool transposeX = gmm::IsTransposeLastTwoDims(params.x1);      // check is transpose x
    bool transposeWeight = gmm::IsTransposeLastTwoDims(params.x2); // check is transpose weight
    // when the last two dims of weight are (1, 1), consider tranB as false
    transposeWeight = transposeWeight && !IsSpecialTranspose(params.x2);
    QGMM_INPLACE_ADD_CHECK_REPORT(
        transposeX == true && transposeWeight == false, return ACLNN_ERR_PARAM_INVALID,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
            QGMM_INPLACE_ADD_ACLNN_OP_NAME, "x1 and x2",
            "x1=" + ViewShapeToString(params.x1) + ", x2=" + ViewShapeToString(params.x2),
            "when transpose of x1 is true and transpose of x2 is false, only this transposition combination is "
            "supported"));

    if (transposeX) {
        CHECK_RET(SetTransViewShape(params.x1, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
        if (params.scale1Optional->GetDataType() == DataType::DT_FLOAT8_E8M0) {
            CHECK_RET(SetTransViewShapeForPertoken(params.scale1Optional, executorPtr) == ACLNN_SUCCESS,
                      ACLNN_ERR_PARAM_INVALID);
        }
    }
    if (transposeWeight) {
        CHECK_RET(SetTransViewShape((params.x2), executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
        if (params.scale2->GetDataType() == DataType::DT_FLOAT8_E8M0) {
            CHECK_RET(SetTransViewShapeForPertoken(params.scale2, executorPtr) == ACLNN_SUCCESS,
                      ACLNN_ERR_PARAM_INVALID);
        }
    }
    CHECK_COND(ParamsDataContiguous(params, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "ParamsDataContiguous failed.");
    // Invoke l0 operator QuantGroupedMatmulInplaceAdd for calculation.
    auto result =
        l0op::QuantGroupedMatmulInplaceAdd(params.x1, params.x2, params.scale1Optional, params.scale2, params.groupList,
                                           params.yRef, params.groupListType, params.groupSize, executorPtr);
    CHECK_RET(result != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // If the output tensor is non-contiguous, convert the calculated contiguous tensor to non-contiguous.
    auto viewCopyResult = l0op::ViewCopy(result, params.yRef, executorPtr);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // Standard syntax, get the size of workspace needed during computation.
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}
} // namespace

#undef QGMM_INPLACE_ADD_CHECK_REPORT

aclnnStatus aclnnQuantGroupedMatmulInplaceAddGetWorkspaceSize(const aclTensor *x1, const aclTensor *x2,
                                                              const aclTensor *scale1Optional, const aclTensor *scale2,
                                                              const aclTensor *groupList, aclTensor *yRef,
                                                              int64_t groupListType, int64_t groupSize,
                                                              uint64_t *workspaceSize, aclOpExecutor **executor)
{
    QGmmInPlaceAdd::QuantGroupedMatmulInplaceAddParams params{x1,        x2,   scale1Optional, scale2,
                                                              groupList, yRef, groupListType,  groupSize};
    // Standard syntax, Check parameters.
    L2_DFX_PHASE_1(aclnnQuantGroupedMatmulInplaceAdd,
                   DFX_IN(x1, x2, scale1Optional, scale2, groupList, yRef, groupListType, groupSize), DFX_OUT(yRef));
    return aclnnQuantGroupedMatmulInplaceAddGetWorkspaceSizeCommon(params, workspaceSize, executor);
}

aclnnStatus aclnnQuantGroupedMatmulInplaceAdd(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                              aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnQuantGroupedMatmulInplaceAdd);
    CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS, ACLNN_ERR_INNER,
               "An error occurred when QuantGMMInplaceAdd launched on aicore.");
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
