/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_batch_matmul_reduce_scatter_all_to_all.h"
#include "common/utils/op_mc2.h"
#include "common/op_host/op_api/mc2_3rd_matmul_util.h"
#include "common/utils/op_mc2_def.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "aclnnInner_batch_mat_mul_reduce_scatter_allto_all.h"
#include "mc2_log_compat.h"
using namespace op;

static const char *BMM_ACLNN_OP_NAME = "aclnnBatchMatMulReduceScatterAlltoAll";

#ifdef __cplusplus
extern "C" {
#endif

static bool CheckNotNull(const aclTensor *x, const aclTensor *weight, const char *groupEp, const char *groupTp,
                         aclTensor *out)
{
    OP_CHECK_NULL(x, return false);
    OP_CHECK_NULL(weight, return false);
    OP_CHECK_NULL(out, return false);
    if ((groupEp == nullptr) || (strnlen(groupEp, HCCL_GROUP_NAME_MAX) == 0)) {
        OP_LOGE_FOR_INVALID_VALUE(BMM_ACLNN_OP_NAME, "groupEp", "empty", "non-empty");
        return false;
    }
    if ((groupTp == nullptr) || (strnlen(groupTp, HCCL_GROUP_NAME_MAX) == 0)) {
        OP_LOGE_FOR_INVALID_VALUE(BMM_ACLNN_OP_NAME, "groupTp", "empty", "non-empty");
        return false;
    }
    return true;
}

// check dtype
static bool CheckDtypeValid(const aclTensor *x, const aclTensor *weight, const aclTensor *bias, const aclTensor *out)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(x, MOE_X_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(weight, MOE_X_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(out, MOE_X_DTYPE_SUPPORT_LIST, return false);

    OP_CHECK_DTYPE_NOT_SAME(x, weight, return false);
    OP_CHECK_DTYPE_NOT_SAME(x, out, return false);

    if (bias != nullptr) { // x和bias支持同时为fp16，或x为bf16和bias为fp32
        OP_CHECK_DTYPE_NOT_SUPPORT(bias, MOE_BIAS_DTYPE_SUPPORT_LIST, return false);
        if (x->GetDataType() == op::DataType::DT_FLOAT16) {
            OP_CHECK_DTYPE_NOT_SAME(bias, x, return false);
        }
        if (x->GetDataType() == op::DataType::DT_BF16) {
            if (bias->GetDataType() != op::DataType::DT_FLOAT) {
                OP_LOGE_FOR_INVALID_DTYPE(BMM_ACLNN_OP_NAME, "bias", op::ToString(bias->GetDataType()).GetString(),
                                          "FLOAT");
                return false;
            }
        }
    }
    return true;
}

// check shape dim
static bool CheckIfTensorThreeDim(const aclTensor *x, const aclTensor *weight, const aclTensor *bias,
                                  const aclTensor *out)
{
    if (x->GetViewShape().GetDimNum() != SUPPORTED_DIMENSIONAL) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            BMM_ACLNN_OP_NAME, "x", (std::to_string(x->GetViewShape().GetDimNum()) + "D").c_str(),
            ("The shape of x must be " + std::to_string(SUPPORTED_DIMENSIONAL) + "D.").c_str());
        return false;
    }
    if (weight->GetViewShape().GetDimNum() != SUPPORTED_DIMENSIONAL) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            BMM_ACLNN_OP_NAME, "weight", (std::to_string(weight->GetViewShape().GetDimNum()) + "D").c_str(),
            ("The shape of weight must be " + std::to_string(SUPPORTED_DIMENSIONAL) + "D.").c_str());
        return false;
    }
    if (out->GetViewShape().GetDimNum() != SUPPORTED_DIMENSIONAL) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            BMM_ACLNN_OP_NAME, "out", (std::to_string(out->GetViewShape().GetDimNum()) + "D").c_str(),
            ("The shape of out must be " + std::to_string(SUPPORTED_DIMENSIONAL) + "D.").c_str());
        return false;
    }
    if (bias != nullptr) {
        if ((bias->GetViewShape().GetDimNum() != SUPPORTED_DIMENSIONAL) &&
            (bias->GetViewShape().GetDimNum() != BIAS_SUPPORTED_DIMENSIONAL)) {
            std::string bmmBiasDimStr = std::to_string(bias->GetViewShape().GetDimNum()) + "D";
            OP_LOGE_FOR_INVALID_SHAPEDIM(BMM_ACLNN_OP_NAME, "bias", bmmBiasDimStr.c_str(), "2D or 3D");
            return false;
        }
    }
    return true;
}

// 维度判断
static bool CheckTensorDimCommonShape(const aclTensor *x, const aclTensor *weight, int64_t epWorldSize,
                                      const aclTensor *out)
{
    // E = E/ep * ep, y_0 == x_0 * ep
    if ((x->GetViewShape().GetDim(DIM_0) * epWorldSize) != out->GetViewShape().GetDim(DIM_0)) {
        std::string bmmShapeStr = "y_0=" + std::to_string(out->GetViewShape().GetDim(DIM_0)) +
                                  ", x_0=" + std::to_string(x->GetViewShape().GetDim(DIM_0)) +
                                  ", ep=" + std::to_string(epWorldSize);
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            BMM_ACLNN_OP_NAME, "x", bmmShapeStr.c_str(),
            "The first dim of x multiplied by epSize must equal the first dim of out");
        return false;
    }
    // x_0 == w_0
    if (x->GetViewShape().GetDim(DIM_0) != weight->GetViewShape().GetDim(DIM_0)) {
        std::string bmmShapeStr = "x_0=" + std::to_string(x->GetViewShape().GetDim(DIM_0)) +
                                  ", W_0=" + std::to_string(weight->GetViewShape().GetDim(DIM_0));
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(BMM_ACLNN_OP_NAME, "x", bmmShapeStr.c_str(),
                                              "The first dim of x must equal the first dim of weight");
        return false;
    }
    // x_2 == w_1
    if (x->GetViewShape().GetDim(DIM_2) != weight->GetViewShape().GetDim(DIM_1)) {
        std::string bmmShapeStr = "x_2=" + std::to_string(x->GetViewShape().GetDim(DIM_2)) +
                                  ", W_1=" + std::to_string(weight->GetViewShape().GetDim(DIM_1));
        OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
            BMM_ACLNN_OP_NAME, "x", bmmShapeStr.c_str(),
            "The last dim of x must equal the second dim of weight (without transpose)");
        return false;
    }
    return true;
}

static bool CheckTensorDimUniqueShape(const aclTensor *x, const aclTensor *weight, int64_t epWorldSize,
                                      int64_t tpWorldSize, int64_t yShardType, const aclTensor *out)
{
    if (yShardType == 0) {
        // H = H/tp * tp, w_2 == y_2 * tp
        if ((out->GetViewShape().GetDim(DIM_2) * tpWorldSize) != weight->GetViewShape().GetDim(DIM_2)) {
            std::string bmmShapeStr = "W_2=" + std::to_string(weight->GetViewShape().GetDim(DIM_2)) +
                                      ", out_1=" + std::to_string(out->GetViewShape().GetDim(DIM_1)) +
                                      ", tp=" + std::to_string(tpWorldSize);
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(BMM_ACLNN_OP_NAME, "weight", bmmShapeStr.c_str(),
                                                  "The shape of weight: H = H/tp * tp, W_2 must equal out_1 * tp");
            return false;
        }
        // x_1 == y_1 * ep
        if ((out->GetViewShape().GetDim(DIM_1) * epWorldSize) != x->GetViewShape().GetDim(DIM_1)) {
            std::string bmmShapeStr = "x_1=" + std::to_string(x->GetViewShape().GetDim(DIM_1)) +
                                      ", out_1=" + std::to_string(out->GetViewShape().GetDim(DIM_1)) +
                                      ", ep=" + std::to_string(epWorldSize);
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(BMM_ACLNN_OP_NAME, "x", bmmShapeStr.c_str(),
                                                  "The shape of x: x_1 must equal out_1 * ep");
            return false;
        }
    } else if (yShardType == 1) {
        // w_2 == y_2
        if (out->GetViewShape().GetDim(DIM_2) != weight->GetViewShape().GetDim(DIM_2)) {
            std::string bmmShapeStr = "out_2=" + std::to_string(out->GetViewShape().GetDim(DIM_2)) +
                                      ", W_2=" + std::to_string(weight->GetViewShape().GetDim(DIM_2));
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(BMM_ACLNN_OP_NAME, "weight", bmmShapeStr.c_str(),
                                                  "The last dim of weight must equal the last dim of out");
            return false;
        }
        // x_1 == y_1 * ep * tp
        if ((out->GetViewShape().GetDim(DIM_1) * epWorldSize * tpWorldSize) != x->GetViewShape().GetDim(DIM_1)) {
            std::string bmmShapeStr = "x_1=" + std::to_string(x->GetViewShape().GetDim(DIM_1)) +
                                      ", out_1=" + std::to_string(out->GetViewShape().GetDim(DIM_1)) +
                                      ", ep=" + std::to_string(epWorldSize) + ", tp=" + std::to_string(tpWorldSize);
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                BMM_ACLNN_OP_NAME, "x", bmmShapeStr.c_str(),
                "The second dim of out multiplied by ep and tp must equal the second dim of x");
            return false;
        }
    } else {
        OP_LOGE_FOR_INVALID_VALUE(BMM_ACLNN_OP_NAME, "yShardType", std::to_string(yShardType).c_str(), "0 or 1");
        return false;
    }
    return true;
}

static bool CheckBiasDim(const aclTensor *weight, const aclTensor *bias, const aclTensor *out)
{
    if (bias != nullptr) {
        if ((bias->GetViewShape().GetDim(DIM_0) != weight->GetViewShape().GetDim(DIM_0)) ||
            ((bias->GetViewShape().GetDimNum() == SUPPORTED_DIMENSIONAL) &&
             (bias->GetViewShape().GetDim(DIM_1) != 1)) ||
            (bias->GetViewShape().GetDim(bias->GetViewShape().GetDimNum() - 1) != out->GetViewShape().GetDim(DIM_2))) {
            std::string bmmBiasShapeStr = op::ToString(bias->GetViewShape()).GetString();
            std::string bmmExpectedShape = "[" + std::to_string(weight->GetViewShape().GetDim(DIM_0)) + "," +
                                           std::to_string(out->GetViewShape().GetDim(DIM_2)) + "] or [" +
                                           std::to_string(weight->GetViewShape().GetDim(DIM_0)) + ",1," +
                                           std::to_string(out->GetViewShape().GetDim(DIM_2)) + "]";
            OP_LOGE_FOR_INVALID_SHAPE(BMM_ACLNN_OP_NAME, "biasOptional", bmmBiasShapeStr.c_str(),
                                      bmmExpectedShape.c_str());
            return false;
        }
    }
    return true;
}

static bool CheckTensorDim(const aclTensor *x, const aclTensor *weight, const aclTensor *bias, int64_t epWorldSize,
                           int64_t tpWorldSize, int64_t yShardType, const aclTensor *out)
{
    // 是否为三维判断
    if (!CheckIfTensorThreeDim(x, weight, bias, out)) {
        return false;
    }

    // 公共shape特性
    if (!CheckTensorDimCommonShape(x, weight, epWorldSize, out)) {
        return false;
    }

    // 非公共shape特性
    if (!CheckTensorDimUniqueShape(x, weight, epWorldSize, tpWorldSize, yShardType, out)) {
        return false;
    }

    // bias 维度判断
    if (!CheckBiasDim(weight, bias, out)) {
        return false;
    }

    return true;
}

// 属性校验
static bool CheckAttr(int64_t epWorldSize, int64_t tpWorldSize, int64_t yShardType)
{
    // yShardType 当前支持 0 或 1
    if ((yShardType) != 0 && (yShardType != 1)) {
        OP_LOGE_FOR_INVALID_VALUE(BMM_ACLNN_OP_NAME, "yShardType", std::to_string(yShardType).c_str(), "0 or 1");
        return false;
    }
    // ep和tp 仅支持2、4、8、16、32
    if ((epWorldSize != WORLD_SIZE_PAIR) && (epWorldSize != WORLD_SIZE_QUAD) && (epWorldSize != WORLD_SIZE_OCTET) &&
        (epWorldSize != WORLD_SIZE_SEXTET) && (epWorldSize != WORLD_SIZE_THIRTYTWO)) {
        OP_LOGE_FOR_INVALID_VALUE(BMM_ACLNN_OP_NAME, "epWorldSize", std::to_string(epWorldSize).c_str(),
                                  "2, 4, 8, 16, or 32");
        return false;
    }
    if ((tpWorldSize != WORLD_SIZE_PAIR) && (tpWorldSize != WORLD_SIZE_QUAD) && (tpWorldSize != WORLD_SIZE_OCTET) &&
        (tpWorldSize != WORLD_SIZE_SEXTET) && (tpWorldSize != WORLD_SIZE_THIRTYTWO)) {
        OP_LOGE_FOR_INVALID_VALUE(BMM_ACLNN_OP_NAME, "tpWorldSize", std::to_string(tpWorldSize).c_str(),
                                  "2, 4, 8, 16, or 32");
        return false;
    }
    return true;
}

// 范围校验
static bool CheckShapeRange(const aclTensor *x, const aclTensor *weight, const aclTensor *out)
{
    // 暂不支持空tensor场景
    // M/tp in [1, 65535], 空tensor K校验
    if ((x->GetViewShape().GetDim(DIM_2) <= 0) || (x->GetViewShape().GetDim(DIM_2) > MATMUL_K_LIMIT)) {
        std::string bmmRangeStr = "M/tp=" + std::to_string(x->GetViewShape().GetDim(DIM_2));
        std::string bmmExpectedStr = "[1, " + std::to_string(MATMUL_K_LIMIT) + "]";
        OP_LOGE_FOR_INVALID_VALUE(BMM_ACLNN_OP_NAME, "M/tp", bmmRangeStr.c_str(), bmmExpectedStr.c_str());
        return false;
    }
    // E/ep in [1, 32], 空tensor E校验
    if ((x->GetViewShape().GetDim(DIM_0) <= 0) || (x->GetViewShape().GetDim(DIM_0) > MATMUL_E_OVER_EP_LIMIT)) {
        std::string bmmRangeStr = "E/ep=" + std::to_string(x->GetViewShape().GetDim(DIM_0));
        std::string bmmExpectedStr = "[1, " + std::to_string(MATMUL_E_OVER_EP_LIMIT) + "]";
        OP_LOGE_FOR_INVALID_VALUE(BMM_ACLNN_OP_NAME, "E/ep", bmmRangeStr.c_str(), bmmExpectedStr.c_str());
        return false;
    }
    // E in [2, 512]
    if ((out->GetViewShape().GetDim(DIM_0) < EXPERT_LOWER_LIMIT) ||
        (out->GetViewShape().GetDim(DIM_0) > EXPERT_UPPER_LIMIT)) {
        std::string bmmRangeStr = "E=" + std::to_string(out->GetViewShape().GetDim(DIM_0));
        std::string bmmExpectedStr =
            "[" + std::to_string(EXPERT_LOWER_LIMIT) + ", " + std::to_string(EXPERT_UPPER_LIMIT) + "]";
        OP_LOGE_FOR_INVALID_VALUE(BMM_ACLNN_OP_NAME, "E", bmmRangeStr.c_str(), bmmExpectedStr.c_str());
        return false;
    }
    // C > 0, C/tp > 0, 空tensor M校验
    if (out->GetViewShape().GetDim(DIM_1) <= 0) {
        std::string bmmRangeStr = "C=" + std::to_string(out->GetViewShape().GetDim(DIM_1));
        OP_LOGE_FOR_INVALID_VALUE(BMM_ACLNN_OP_NAME, "C", bmmRangeStr.c_str(), "positive");
        return false;
    }
    // H in [1, 65535], 空tensor N校验
    if ((weight->GetViewShape().GetDim(DIM_2) <= 0) || (weight->GetViewShape().GetDim(DIM_2) > MATMUL_K_LIMIT)) {
        std::string bmmRangeStr = "H=" + std::to_string(weight->GetViewShape().GetDim(DIM_2));
        std::string bmmExpectedStr = "[1, " + std::to_string(MATMUL_K_LIMIT) + "]";
        OP_LOGE_FOR_INVALID_VALUE(BMM_ACLNN_OP_NAME, "H", bmmRangeStr.c_str(), bmmExpectedStr.c_str());
        return false;
    }
    return true;
}

// 入参校验
static aclnnStatus CheckParams(const aclTensor *x, const aclTensor *weight, const aclTensor *biasOptional,
                               const char *groupEp, const char *groupTp, int64_t epWorldSize, int64_t tpWorldSize,
                               int64_t yShardType, aclTensor *out)
{
    CHECK_RET(CheckNotNull(x, weight, groupEp, groupTp, out), ACLNN_ERR_PARAM_NULLPTR);

    if (strnlen(groupEp, HCCL_GROUP_NAME_MAX) >= HCCL_GROUP_NAME_MAX) {
        OP_LOGE_WITH_INVALID_ATTR_SIZE(BMM_ACLNN_OP_NAME, "groupEp",
                                       std::to_string(strnlen(groupEp, HCCL_GROUP_NAME_MAX)).c_str(),
                                       std::to_string(HCCL_GROUP_NAME_MAX).c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (strnlen(groupTp, HCCL_GROUP_NAME_MAX) >= HCCL_GROUP_NAME_MAX) {
        OP_LOGE_WITH_INVALID_ATTR_SIZE(BMM_ACLNN_OP_NAME, "groupTp",
                                       std::to_string(strnlen(groupTp, HCCL_GROUP_NAME_MAX)).c_str(),
                                       std::to_string(HCCL_GROUP_NAME_MAX).c_str());
        return ACLNN_ERR_PARAM_INVALID;
    }

    CHECK_RET(CheckDtypeValid(x, weight, biasOptional, out), ACLNN_ERR_PARAM_INVALID);

    CHECK_RET(CheckAttr(epWorldSize, tpWorldSize, yShardType), ACLNN_ERR_PARAM_INVALID);

    CHECK_RET(CheckTensorDim(x, weight, biasOptional, epWorldSize, tpWorldSize, yShardType, out),
              ACLNN_ERR_PARAM_INVALID);

    CHECK_RET(CheckShapeRange(x, weight, out), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static const aclTensor *TransTensor(const aclTensor *x2)
{
    uint64_t storageDimsNum = x2->GetStorageShape().GetDimNum();
    std::vector<int64_t> storageDims(storageDimsNum);
    for (uint64_t i = 0; i < storageDimsNum; i++) {
        storageDims[i] = x2->GetStorageShape().GetDim(i);
    }

    uint64_t viewDimsNum = x2->GetViewShape().GetDimNum();
    std::vector<int64_t> viewDims;
    viewDims.resize(viewDimsNum);
    for (uint64_t i = 0; i < viewDimsNum; i++) {
        viewDims[i] = x2->GetViewShape().GetDim(i);
    }
    viewDims[DIM_0] = x2->GetViewShape().GetDim(DIM_0);
    viewDims[DIM_1] = x2->GetViewShape().GetDim(DIM_2); // transpose the last two dimensions
    viewDims[DIM_2] = x2->GetViewShape().GetDim(DIM_1);

    aclDataType dataType = aclDataType::ACL_DT_UNDEFINED;
    aclGetDataType(x2, &dataType);
    auto stride = x2->GetViewStrides();

    std::vector<int64_t> strideCopy(viewDimsNum);
    strideCopy[DIM_0] = stride[DIM_0];
    strideCopy[DIM_1] = stride[DIM_2]; // transpose the last two dimensions
    strideCopy[DIM_2] = stride[DIM_1];

    auto offset = x2->GetViewOffset();
    aclFormat format = aclFormat::ACL_FORMAT_ND;

    return aclCreateTensor(viewDims.data(), viewDimsNum, dataType, strideCopy.data(), offset, format,
                           storageDims.data(), storageDimsNum, x2->GetTensor()->GetAddr());
}

aclnnStatus aclnnBatchMatMulReduceScatterAlltoAllGetWorkspaceSize(const aclTensor *x, const aclTensor *weight,
                                                                  const aclTensor *biasOptional, const char *groupEp,
                                                                  const char *groupTp, int64_t epWorldSize,
                                                                  int64_t tpWorldSize, int64_t yShardType,
                                                                  aclTensor *out, uint64_t *workspaceSize,
                                                                  aclOpExecutor **executor)
{
    auto ret_param = CheckParams(x, weight, biasOptional, groupEp, groupTp, epWorldSize, tpWorldSize, yShardType, out);
    CHECK_RET(ret_param == ACLNN_SUCCESS, ret_param);

    bool transposeWeight = Ops::Transformer::IsTransposeLastTwoDims(weight);
    if (transposeWeight) {
        if (weight->GetTensor() == nullptr) {
            OP_LOGE_WITH_INVALID_INPUT(BMM_ACLNN_OP_NAME, "weight");
            return ACLNN_ERR_INNER_NULLPTR;
        }
        auto weightTrans = TransTensor(weight);
        aclnnStatus ret = aclnnInnerBatchMatMulReduceScatterAlltoAllGetWorkspaceSize(
            x, weightTrans, biasOptional, const_cast<char *>(groupEp), const_cast<char *>(groupTp), epWorldSize,
            tpWorldSize, yShardType, transposeWeight, out, workspaceSize, executor);
        return ret;
    } else {
        aclnnStatus ret = aclnnInnerBatchMatMulReduceScatterAlltoAllGetWorkspaceSize(
            x, weight, biasOptional, const_cast<char *>(groupEp), const_cast<char *>(groupTp), epWorldSize, tpWorldSize,
            yShardType, transposeWeight, out, workspaceSize, executor);
        return ret;
    }
}

aclnnStatus aclnnBatchMatMulReduceScatterAlltoAll(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                  aclrtStream stream)
{
    aclnnStatus ret = aclnnInnerBatchMatMulReduceScatterAlltoAll(workspace, workspaceSize, executor, stream);
    return ret;
}

#ifdef __cplusplus
}
#endif
