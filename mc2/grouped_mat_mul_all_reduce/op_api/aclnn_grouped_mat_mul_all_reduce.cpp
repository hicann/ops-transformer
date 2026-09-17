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
 * \file aclnn_grouped_mat_mul_all_reduce.cpp
 * \brief
 */
#include "aclnn_grouped_mat_mul_all_reduce.h"
#include <new>
#include "securec.h"

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
#include "mc2_log_compat.h"
#include "common/utils/hccl_util.h"
#include "aclnnInner_grouped_mat_mul_all_reduce.h"

static constexpr int64_t MAX_GROUP_LIST_SIZE = 64; // tiling data size only support 8192 bytes.

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

static const int64_t X_Y_SEPARATED = 0; // x,y不切?
static const int64_t Y_SEPARATED = 1;   // x切分
static const int64_t X_SEPARATED = 2;   // y切分
static const int64_t NO_SEPARATED = 3;  // x,y切分

static const size_t MAX_FM_DIM = 6;
static const size_t MIN_FM_DIM = 2;
static const size_t SPLIT_DIM = 2;

const std::map<DataType, aclDataType> BIAS_DTYPE{{DataType::DT_FLOAT16, aclDataType::ACL_FLOAT16},
                                                 {DataType::DT_BF16, aclDataType::ACL_FLOAT}};

struct NnopbaseDfxId {
    uint32_t id;
    const char *funcName;
    bool hasReg;
};

extern uint64_t NnopbaseMsprofSysTime();
extern void NnopbaseReportApiInfo(const uint64_t beginTime, NnopbaseDfxId &dfxId);

static void SplitTensorList(const std::vector<std::vector<int64_t>> &shapes, const aclTensorList *tensorList,
                            std::vector<aclTensor *> &tensors)
{
    uint8_t *addr = static_cast<uint8_t *>((*tensorList)[0]->GetStorageAddr());
    aclDataType dataType;
    aclGetDataType((*tensorList)[0], &dataType);
    aclFormat format;
    aclGetFormat((*tensorList)[0], &format);
    uint64_t groupListSum = 0;
    for (size_t i = 0; i < shapes.size(); i++) {
        if (shapes[i].empty()) {
            aclTensor *tensor = nullptr;
            tensors.emplace_back(tensor);
            continue;
        }
        aclTensor *tensor = aclCreateTensor(
            shapes[i].data(), shapes[i].size(), dataType, nullptr, 0, format, shapes[i].data(), shapes[i].size(),
            addr + groupListSum * shapes[i][1] * TypeSize((*tensorList)[0]->GetDataType()));
        groupListSum += shapes[i][0];
        tensors.emplace_back(tensor);
    }
}

struct GroupedMatMulAllReduceParams {
    const aclTensorList *x = nullptr;
    const aclTensorList *weight = nullptr;
    const aclTensorList *bias = nullptr;
    const aclIntArray *groupListOptional = nullptr;
    int64_t splitItemOptional;
    const char *group = nullptr;
    const char *reduceOp = nullptr; // reduce type, now 'sum'
    int64_t commTurn;               // communication trun num
    const aclTensorList *y = nullptr;
    DataType xDtype;
};

static bool CheckNotNull(const GroupedMatMulAllReduceParams &gmmParams)
{
    CHECK_COND(gmmParams.x != nullptr, false, "x must not be nullptr.");
    CHECK_COND(gmmParams.weight != nullptr, false, "weight must not be nullptr.");
    CHECK_COND(gmmParams.y != nullptr, false, "y must not be nullptr.");
    return true;
}

static bool CheckGroupListOptional(const GroupedMatMulAllReduceParams &gmmParams)
{
    CHECK_COND(gmmParams.groupListOptional != nullptr, false,
               "When splitItem is 1/3, groupList size must not be nullptr.");
    uint64_t groupListSize = gmmParams.groupListOptional->Size();
    CHECK_COND(groupListSize == gmmParams.weight->Size(), false,
               "When splitItem is 1/3, groupList size[%lu] must be equal with weight group size[%lu].", groupListSize,
               gmmParams.weight->Size());
    int64_t preGoupList = 0;
    for (size_t i = 0; i < groupListSize; i++) {
        CHECK_COND((*gmmParams.groupListOptional)[i] >= preGoupList, false,
                   "groupListOptional should be an incremental sequence.");
        preGoupList = (*gmmParams.groupListOptional)[i];
    }
    CHECK_COND((*gmmParams.x)[0]->GetViewShape().GetDim(0) == preGoupList, false,
               "The last value of group list(%lu) must equal with x shape[0] (%lu).", preGoupList,
               (*gmmParams.x)[0]->GetViewShape().GetDim(0));
    return true;
}

static aclnnStatus CheckInputGroupSizes(const GroupedMatMulAllReduceParams &gmmParams, uint64_t xGroupedSize,
                                        uint64_t weightGroupedSize)
{
    if (gmmParams.splitItemOptional == Y_SEPARATED || gmmParams.splitItemOptional == NO_SEPARATED) {
        if (xGroupedSize != 1) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("aclnnGroupedMatMulAllReduce", "x group size",
                                                  std::to_string(xGroupedSize).c_str(),
                                                  "When splitItemOptional is 1/3, x group size must be 1");
            return ACLNN_ERR_PARAM_INVALID;
        }
        CHECK_RET(CheckGroupListOptional(gmmParams), ACLNN_ERR_PARAM_INVALID);
    } else if (gmmParams.splitItemOptional == X_SEPARATED || gmmParams.splitItemOptional == X_Y_SEPARATED) {
        if (xGroupedSize != weightGroupedSize) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("aclnnGroupedMatMulAllReduce", "x group size",
                                                  std::to_string(xGroupedSize).c_str(),
                                                  "When splitItem is 0/2, x group size must equal weight group size " +
                                                      std::to_string(weightGroupedSize));
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (gmmParams.groupListOptional != nullptr) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("aclnnGroupedMatMulAllReduce", "groupListOptional", "not null",
                                                  "When splitItem is 0/2, groupList must be nullptr");
            return ACLNN_ERR_PARAM_INVALID;
        }
    } else {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("aclnnGroupedMatMulAllReduce", "splitItemOptional",
                                              std::to_string(gmmParams.splitItemOptional).c_str(),
                                              "only support 0/1/2/3");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckBiasGroupDimensions(const GroupedMatMulAllReduceParams &gmmParams, uint64_t weightGroupedSize)
{
    if (gmmParams.bias != nullptr) {
        uint64_t biasGroupedSize = gmmParams.bias->Size();
        if (weightGroupedSize != biasGroupedSize) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                "aclnnGroupedMatMulAllReduce", "bias group size", std::to_string(biasGroupedSize).c_str(),
                "should equal weight group size " + std::to_string(weightGroupedSize));
            return ACLNN_ERR_PARAM_INVALID;
        }
        for (size_t i = 0; i < biasGroupedSize; ++i) {
            OP_CHECK_NULL((*gmmParams.bias)[i], continue);
            size_t biasDims = (*gmmParams.bias)[i]->GetViewShape().GetDimNum();
            if (biasDims != 1) {
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("aclnnGroupedMatMulAllReduce",
                                                         ("bias[" + std::to_string(i) + "]").c_str(),
                                                         std::to_string(biasDims).c_str(), "Dim must be 1");
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckInputGroupDimensions(const GroupedMatMulAllReduceParams &gmmParams, uint64_t xGroupedSize,
                                             uint64_t weightGroupedSize)
{
    for (size_t i = 0; i < xGroupedSize; ++i) {
        OP_CHECK_NULL((*gmmParams.x)[i], continue);
        size_t xDims = (*gmmParams.x)[i]->GetViewShape().GetDimNum();
        if (xDims > MAX_FM_DIM || xDims < MIN_FM_DIM) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("aclnnGroupedMatMulAllReduce",
                                                     ("x[" + std::to_string(i) + "]").c_str(),
                                                     std::to_string(xDims).c_str(), "Dim only support 2-6");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    for (size_t i = 0; i < weightGroupedSize; ++i) {
        OP_CHECK_NULL((*gmmParams.weight)[i], continue);
        size_t weightDims = (*gmmParams.weight)[i]->GetViewShape().GetDimNum();
        if (weightDims != MIN_FM_DIM) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("aclnnGroupedMatMulAllReduce",
                                                     ("weight[" + std::to_string(i) + "]").c_str(),
                                                     std::to_string(weightDims).c_str(), "Dim only support 2");
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckParamDimAndLengthGmmAr(const GroupedMatMulAllReduceParams &gmmParams)
{
    uint64_t xGroupedSize = gmmParams.x->Size();
    uint64_t weightGroupedSize = gmmParams.weight->Size();
    uint64_t yGroupedSize = gmmParams.y->Size();
    if (weightGroupedSize > MAX_GROUP_LIST_SIZE) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("aclnnGroupedMatMulAllReduce", "weight group size",
                                              std::to_string(weightGroupedSize).c_str(),
                                              "should not exceed " + std::to_string(MAX_GROUP_LIST_SIZE));
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (CheckInputGroupSizes(gmmParams, xGroupedSize, weightGroupedSize) != ACLNN_SUCCESS) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (gmmParams.splitItemOptional == X_SEPARATED || gmmParams.splitItemOptional == NO_SEPARATED) {
        if (yGroupedSize != 1) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("aclnnGroupedMatMulAllReduce", "y group size",
                                                  std::to_string(yGroupedSize).c_str(),
                                                  "When splitItemOptional is 2/3, y group size must be 1");
            return ACLNN_ERR_PARAM_INVALID;
        }
    } else {
        if (yGroupedSize != weightGroupedSize) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                "aclnnGroupedMatMulAllReduce", "y group size", std::to_string(yGroupedSize).c_str(),
                "When splitItemOptional is 0/1, y group size must equal weight group size " +
                    std::to_string(weightGroupedSize));
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (CheckInputGroupDimensions(gmmParams, xGroupedSize, weightGroupedSize) != ACLNN_SUCCESS) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (CheckBiasGroupDimensions(gmmParams, weightGroupedSize) != ACLNN_SUCCESS) {
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckDimK(const GroupedMatMulAllReduceParams &gmmParams)
{
    bool isXSeparated = gmmParams.splitItemOptional == X_Y_SEPARATED || gmmParams.splitItemOptional == X_SEPARATED;
    auto xTensor = (*gmmParams.x)[0]; // 0: first x tensor
    if (xTensor == nullptr) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("aclnnGroupedMatMulAllReduce", "x[0]", "nullptr",
                                              "x[0] cannot be an empty tensor.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    for (size_t i = 0; i < gmmParams.weight->Size(); ++i) {
        if (isXSeparated) {
            xTensor = (*gmmParams.x)[i];
            if (xTensor == nullptr) {
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("aclnnGroupedMatMulAllReduce",
                                                      ("x[" + std::to_string(i) + "]").c_str(), "nullptr",
                                                      "When splitItem = 0/2, x cannot be empty");
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
        auto wTensor = (*gmmParams.weight)[i];
        size_t xDims = xTensor->GetViewShape().GetDimNum();
        int64_t xDimLast = xTensor->GetViewShape().GetDim(xDims - 1);
        if (wTensor == nullptr) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("aclnnGroupedMatMulAllReduce",
                                                  ("w[" + std::to_string(i) + "]").c_str(), "nullptr",
                                                  "should not be empty");
            return ACLNN_ERR_PARAM_INVALID;
        }
        int64_t wDimFirst = wTensor->GetViewShape().GetDim(0);
        if (xDimLast != wDimFirst) {
            std::string reason = "The last dim of x = " + std::to_string(xDimLast) +
                                 " is not equal to the first dim of weight = " + std::to_string(wDimFirst);
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                "aclnnGroupedMatMulAllReduce",
                ("x[" + std::to_string(i) + "]/weight[" + std::to_string(i) + "]").c_str(),
                ("[" + std::to_string(xDimLast) + "," + std::to_string(wDimFirst) + "]").c_str(), reason.c_str());
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckSeparatedXYDimensions(const GroupedMatMulAllReduceParams &gmmParams)
{
    for (size_t i = 0; i < gmmParams.x->Size(); ++i) {
        auto xTensor = (*gmmParams.x)[i];
        auto yTensor = (*gmmParams.y)[i];
        if (xTensor == nullptr) {
            if (yTensor != nullptr) {
                OP_LOGE_WITH_INVALID_INPUT("aclnnGroupedMatMulAllReduce", ("x[" + std::to_string(i) + "]").c_str());
                return ACLNN_ERR_PARAM_INVALID;
            }
            continue;
        } else {
            if (yTensor == nullptr) {
                OP_LOGE_WITH_INVALID_INPUT("aclnnGroupedMatMulAllReduce", ("y[" + std::to_string(i) + "]").c_str());
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
        size_t xDims = xTensor->GetViewShape().GetDimNum();
        size_t yDims = yTensor->GetViewShape().GetDimNum();
        if (xDims != yDims || xDims >= MAX_FM_DIM) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                "aclnnGroupedMatMulAllReduce", ("x[" + std::to_string(i) + "]/y[" + std::to_string(i) + "]").c_str(),
                ("x:" + std::to_string(xDims) + "D, y:" + std::to_string(yDims) + "D").c_str(),
                "When splitItem is 0, x dims must equal y dims and not greater than " + std::to_string(MAX_FM_DIM));
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckDims(const GroupedMatMulAllReduceParams &gmmParams)
{
    if (gmmParams.splitItemOptional == X_Y_SEPARATED) {
        if (CheckSeparatedXYDimensions(gmmParams) != ACLNN_SUCCESS) {
            return ACLNN_ERR_PARAM_INVALID;
        }
    } else {
        for (size_t i = 0; i < gmmParams.x->Size(); i++) {
            auto xTensor = (*gmmParams.x)[i];
            if (xTensor != nullptr) {
                auto xDims = xTensor->GetViewShape().GetDimNum();
                if (xDims != MIN_FM_DIM) {
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        "aclnnGroupedMatMulAllReduce", ("x[" + std::to_string(i) + "]").c_str(),
                        std::to_string(xDims).c_str(),
                        "When splitItem is not 0, x dims must be " + std::to_string(MIN_FM_DIM));
                    return ACLNN_ERR_PARAM_INVALID;
                }
            }
        }
        for (size_t i = 0; i < gmmParams.y->Size(); i++) {
            auto yTensor = (*gmmParams.y)[i];
            if (yTensor != nullptr) {
                auto yDims = yTensor->GetViewShape().GetDimNum();
                if (yDims != MIN_FM_DIM) {
                    OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
                        "aclnnGroupedMatMulAllReduce", ("y[" + std::to_string(i) + "]").c_str(),
                        std::to_string(yDims).c_str(),
                        "When splitItem is not 0, y dims must be " + std::to_string(MIN_FM_DIM));
                    return ACLNN_ERR_PARAM_INVALID;
                }
            }
        }
    }

    if (CheckDimK(gmmParams) != ACLNN_SUCCESS) {
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON("aclnnGroupedMatMulAllReduce", "x/weight", "mismatch",
                                              "K dim of x tensors and weight tensors does match");
        return ACLNN_ERR_PARAM_INVALID;
    }

    return ACLNN_SUCCESS;
}

static bool CheckMatmulShape(const GroupedMatMulAllReduceParams &gmmParams)
{
    for (size_t i = 0; i < gmmParams.x->Size(); ++i) {
        if ((*gmmParams.weight)[i] == nullptr) {
            OP_LOGE_LIBOPAPI_REPORT("aclnnGroupedMatMulAllReduce", "weight[%zu] cannot be nullptr.", i);
            return false;
        }
        if ((*gmmParams.x)[i] == nullptr && (*gmmParams.y)[i] == nullptr) {
            continue;
        }
        if ((*gmmParams.x)[i] == nullptr || (*gmmParams.y)[i] == nullptr) {
            OP_LOGE_LIBOPAPI_REPORT("aclnnGroupedMatMulAllReduce", "x or y is nullptr.");
            return false;
        }
        op::Shape xShape = (*gmmParams.x)[i]->GetViewShape();
        op::Shape yShape = (*gmmParams.y)[i]->GetViewShape();
        size_t checkDim = xShape.GetDimNum() - 1;
        for (size_t dim = 0; dim < checkDim; dim++) {
            int64_t xMDimValue = xShape.GetDim(dim);
            int64_t yMDimValue = yShape.GetDim(dim);
            CHECK_COND(xMDimValue == yMDimValue, false,
                       "Dim[%zu] of x[%zu] and y[%zu] are %ld and %ld, respectively, "
                       "which should be equal to each other.",
                       dim, i, i, xMDimValue, yMDimValue);
        }
        int64_t xKDimValue = (*gmmParams.x)[i]->GetViewShape().GetDim(checkDim);
        int64_t weightKDimValue = (*gmmParams.weight)[i]->GetViewShape().GetDim(0);
        int64_t weightNDimValue = (*gmmParams.weight)[i]->GetViewShape().GetDim(1);
        int64_t yNDimValue = (*gmmParams.y)[i]->GetViewShape().GetDim(checkDim);
        CHECK_COND(xKDimValue == weightKDimValue, false,
                   "KDim[%ld] of weight[%zu] should be equal with "
                   "KDim[%ld] of x[%zu].",
                   weightKDimValue, i, xKDimValue, i);
        CHECK_COND(weightNDimValue == yNDimValue, false,
                   "NDim[%ld] of weight[%zu] should be equal with "
                   "NDim[%ld] of y[%zu].",
                   weightNDimValue, i, yNDimValue, i);
        if (gmmParams.bias == nullptr) {
            continue;
        }
        CHECK_COND((*gmmParams.bias)[i] != nullptr, false, "bias[%ld] should not be nullptr.", i);
        int64_t biasDimValue = (*gmmParams.bias)[i]->GetViewShape().GetDim(0);
        CHECK_COND(biasDimValue == weightNDimValue, false,
                   "NDim[%ld] of weight[%zu] should be equal with "
                   "NDim[%ld] of bias[%zu].",
                   weightNDimValue, i, biasDimValue, i);
    }
    return true;
}

static bool CheckTensorListDataType(const aclTensorList *tensorList, const DataType dtype)
{
    for (size_t i = 0; i < tensorList->Size(); ++i) {
        const aclTensor *tensor = (*tensorList)[i];
        OP_CHECK_NULL(tensor, continue);
        OP_CHECK_DTYPE_NOT_MATCH(tensor, dtype, return false);
    }
    return true;
}

static bool CheckMatmulDataType(const GroupedMatMulAllReduceParams &gmmParams, const DataType comDtype,
                                const DataType weightDtype, const DataType biasDtype)
{
    if (!CheckTensorListDataType(gmmParams.x, comDtype)) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON("aclnnGroupedMatMulAllReduce", "x", op::ToString(comDtype).GetString(),
                                              "x dtype does not match with required dtype");
        return false;
    }
    if (!CheckTensorListDataType(gmmParams.weight, weightDtype)) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON("aclnnGroupedMatMulAllReduce", "weight",
                                              op::ToString(weightDtype).GetString(),
                                              "weight dtype does not match with required dtype");
        return false;
    }
    if (!CheckTensorListDataType(gmmParams.y, comDtype)) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON("aclnnGroupedMatMulAllReduce", "y", op::ToString(comDtype).GetString(),
                                              "y dtype does not match with required dtype");
        return false;
    }
    if (gmmParams.bias != nullptr) {
        if (!CheckTensorListDataType(gmmParams.bias, biasDtype)) {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON("aclnnGroupedMatMulAllReduce", "bias",
                                                  op::ToString(biasDtype).GetString(),
                                                  "bias dtype does not match with required dtype");
            return false;
        }
    }
    return true;
}

static aclnnStatus CheckParamOptionGmmAr(const GroupedMatMulAllReduceParams &gmmParams)
{
    DataType weightDtype = (*gmmParams.weight)[0]->GetDataType();
    if ((gmmParams.xDtype == DataType::DT_BF16 || gmmParams.xDtype == DataType::DT_FLOAT16) &&
        gmmParams.xDtype == weightDtype) {
        DataType biasDtype = gmmParams.xDtype == DataType::DT_BF16 ? DataType::DT_FLOAT : DataType::DT_FLOAT16;
        CHECK_RET(CheckMatmulDataType(gmmParams, gmmParams.xDtype, weightDtype, biasDtype), ACLNN_ERR_PARAM_INVALID);
    } else {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON("aclnnGroupedMatMulAllReduce", "x/weight",
                                              op::ToString(gmmParams.xDtype).GetString(),
                                              "Only float16/bfloat16 are supported for x and weight.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckParam(const GroupedMatMulAllReduceParams &gmmParams)
{
    CHECK_RET(CheckNotNull(gmmParams), ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(CheckParamDimAndLengthGmmAr(gmmParams) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckParamOptionGmmAr(gmmParams) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckDims(gmmParams) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static void InitShape(const GroupedMatMulAllReduceParams &gmmParams, const bool isXTensor,
                      std::vector<std::vector<int64_t>> &shapes)
{
    shapes.reserve(gmmParams.weight->Size());
    int64_t preOffset = 0;
    for (uint64_t i = 0; i < gmmParams.weight->Size(); i++) {
        std::vector<int64_t> shape;
        if ((*gmmParams.weight)[i] == nullptr) {
            shapes.emplace_back(shape);
            continue;
        }
        shape.reserve(SPLIT_DIM);
        if (isXTensor) {
            shape.emplace_back((*gmmParams.groupListOptional)[i] - preOffset);
            preOffset = (*gmmParams.groupListOptional)[i];
            shape.emplace_back((*gmmParams.weight)[i]->GetViewShape().GetDim(0));
        } else {
            shape.emplace_back((*gmmParams.x)[i]->GetViewShape().GetDim(0));
            shape.emplace_back((*gmmParams.weight)[i]->GetViewShape().GetDim(1));
        }
        shapes.emplace_back(shape);
    }
}

static void CheckOptionalTensorListEmpty(const aclTensorList *&tensorList)
{
    if (tensorList != nullptr) {
        if (tensorList->Size() == 0) {
            tensorList = nullptr;
        } else if ((*tensorList)[0] == nullptr) {
            tensorList = nullptr;
        } else if (tensorList->Size() == 1 && (*tensorList)[0]->GetViewShape().GetDim(0) == 0) {
            tensorList = nullptr;
        }
    }
    return;
}

static void ResetEmptyTensor(GroupedMatMulAllReduceParams &gmmParams)
{
    if (gmmParams.groupListOptional != nullptr && gmmParams.groupListOptional->Size() == 0) {
        gmmParams.groupListOptional = nullptr;
    }
    CheckOptionalTensorListEmpty(gmmParams.bias);
    return;
}

static aclnnStatus CreateXYTensorList(const aclTensorList *xAfterSplit, const aclTensorList *yAfterSplit,
                                      GroupedMatMulAllReduceParams &gmmParams)
{
    if (gmmParams.splitItemOptional == Y_SEPARATED || gmmParams.splitItemOptional == NO_SEPARATED) {
        std::vector<std::vector<int64_t>> shapes;
        InitShape(gmmParams, true, shapes);
        std::vector<aclTensor *> xTensorList;
        SplitTensorList(shapes, gmmParams.x, xTensorList);
        xAfterSplit = aclCreateTensorList(xTensorList.data(), xTensorList.size());
        gmmParams.x = xAfterSplit;
    }
    if (gmmParams.splitItemOptional == X_SEPARATED || gmmParams.splitItemOptional == NO_SEPARATED) {
        std::vector<std::vector<int64_t>> shapes;
        InitShape(gmmParams, false, shapes);
        std::vector<aclTensor *> yTensorList;
        SplitTensorList(shapes, gmmParams.y, yTensorList);
        yAfterSplit = aclCreateTensorList(yTensorList.data(), yTensorList.size());
        gmmParams.y = yAfterSplit;
    }
    return ACLNN_SUCCESS;
}

static void CreateEmptyTensor(const aclDataType dataType, const aclTensorList *&gmmTensorList,
                              aclTensorList *&tensorList)
{
    if (gmmTensorList == nullptr) {
        std::vector<aclTensor *> emptyTensors;
        aclTensor *emptyTensor = aclCreateTensor({}, 0, dataType, nullptr, 0, aclFormat::ACL_FORMAT_ND, {}, 0, nullptr);
        emptyTensors.emplace_back(emptyTensor);
        tensorList = aclCreateTensorList(emptyTensors.data(), emptyTensors.size());
        gmmTensorList = tensorList;
    }
    return;
}

static aclnnStatus PreAndPostProcessForInner(GroupedMatMulAllReduceParams &gmmParams, uint64_t *workspaceSize,
                                             aclOpExecutor **executor)
{
    aclTensorList *emptyBiasList = nullptr; // init emptyBiasList
    if (BIAS_DTYPE.find(gmmParams.xDtype) == BIAS_DTYPE.cend()) {
        OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON("aclnnGroupedMatMulAllReduce", "bias",
                                              op::ToString(gmmParams.xDtype).GetString(),
                                              "Cannot find bias dtype match with xDtype");
        return ACLNN_ERR_PARAM_INVALID;
    }
    CreateEmptyTensor(BIAS_DTYPE.at(gmmParams.xDtype), gmmParams.bias, emptyBiasList);
    aclnnStatus ret = aclnnInnerGroupedMatMulAllReduceGetWorkspaceSize(
        gmmParams.x, gmmParams.weight, gmmParams.bias, gmmParams.groupListOptional, gmmParams.splitItemOptional,
        const_cast<char *>(gmmParams.group), const_cast<char *>(gmmParams.reduceOp), gmmParams.commTurn, gmmParams.y,
        workspaceSize, executor);

    if (emptyBiasList != nullptr) { // destroy tensorList
        aclDestroyTensorList(emptyBiasList);
    }
    return ret;
}

aclnnStatus aclnnGroupedMatMulAllReduceGetWorkspaceSize(const aclTensorList *x, const aclTensorList *weight,
                                                        const aclTensorList *bias, const aclIntArray *groupListOptional,
                                                        int64_t splitItem, const char *group, const char *reduceOp,
                                                        int64_t commTurn, int64_t streamMode, const aclTensorList *y,
                                                        uint64_t *workspaceSize, aclOpExecutor **executor)
{
    uint64_t timeStamp = NnopbaseMsprofSysTime();
    if (streamMode != 1) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("aclnnGroupedMatMulAllReduce", "streamMode",
                                              std::to_string(streamMode).c_str(), "only support 1");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (commTurn != 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("aclnnGroupedMatMulAllReduce", "commTurn",
                                              std::to_string(commTurn).c_str(), "only support 0");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (strcmp(reduceOp, "sum") != 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("aclnnGroupedMatMulAllReduce", "reduceOp", reduceOp, "only support sum");
        return ACLNN_ERR_PARAM_INVALID;
    }
    DataType xDtype = DataType::DT_UNDEFINED;
    for (size_t i = 0; i < x->Size(); ++i) {
        if ((*x)[i] != nullptr) {
            xDtype = (*x)[i]->GetDataType();
            break;
        }
    }
    GroupedMatMulAllReduceParams gmmParams{x,        weight, bias,  groupListOptional, splitItem, group, reduceOp,
                                           commTurn, y,      xDtype};
    ResetEmptyTensor(gmmParams);
    CHECK_RET(CheckParam(gmmParams) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    aclTensorList *xAfterSplit = nullptr;
    aclTensorList *yAfterSplit = nullptr;
    CreateXYTensorList(xAfterSplit, yAfterSplit, gmmParams);
    CHECK_RET(CheckMatmulShape(gmmParams), ACLNN_ERR_PARAM_INVALID);

    // GetWorkspaceSize
    aclnnStatus ret = PreAndPostProcessForInner(gmmParams, workspaceSize, executor);
    OP_LOGD("GroupedMatMulAllReduce, end ret %d.", ret);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    if (xAfterSplit != nullptr) {
        aclDestroyTensorList(xAfterSplit);
    }
    if (yAfterSplit != nullptr) {
        aclDestroyTensorList(yAfterSplit);
    }

    static NnopbaseDfxId dfxId = {0x60000, __func__, false};
    NnopbaseReportApiInfo(timeStamp, dfxId);
    return ret;
}

aclnnStatus aclnnGroupedMatMulAllReduce(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                        aclrtStream stream)
{
    uint64_t timeStamp = NnopbaseMsprofSysTime();
    // Kernel Func
    if (aclnnInnerGroupedMatMulAllReduce(workspace, workspaceSize, executor, stream) != ACLNN_SUCCESS) {
        OP_LOGE_LIBOPAPI_REPORT("aclnnGroupedMatMulAllReduce",
                                "GroupedMatMulAllReduce, This is an error in launch aicore");
        return ACLNN_ERR_INNER;
    }
    static NnopbaseDfxId dfxId = {0x60000, __func__, false};
    NnopbaseReportApiInfo(timeStamp, dfxId);
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
