/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_GROUPED_MATMUL_FINALIZE_ROUTING_WEIGHT_QUANT_950_CHECKER_H
#define OP_API_INC_GROUPED_MATMUL_FINALIZE_ROUTING_WEIGHT_QUANT_950_CHECKER_H
#include "opdev/format_utils.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "quant_grouped_matmul_finalize_routing_util.h"
#include "../../grouped_matmul/op_api/grouped_matmul_util.h"
#include "util/math_util.h"
#include "log/log.h"

#define OP_CHECK_DTYPE_NOT_SUPPORT_WITH_REASON(aclnnName, tensor, supportList, retExpr) \
    do { \
        if (!CheckType(tensor->GetDataType(), supportList)) { \
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON( \
                aclnnName, #tensor, op::ToString(tensor->GetDataType()).GetString(), \
                "The dtype of " + std::string(#tensor) + " must be one of " + op::ToString(supportList).GetString()); \
            retExpr; \
        } \
    } while (0)

namespace {
static constexpr const char *ACLNN_NAME = "aclnnGroupedMatmulFinalizeRoutingWeightNzV2GetWorkspaceSize";
};

namespace GmmFinalizeRouting {

constexpr size_t WQ_DIM_ZERO = 0UL;
constexpr size_t WQ_DIM_ONE = 1UL;
constexpr size_t WQ_DIM_TWO = 2UL;
constexpr size_t WQ_DIM_THREE = 3UL;
constexpr size_t WQ_DIM_FOUR = 4UL;
constexpr size_t WQ_DIM_FIVE = 5UL;
constexpr int64_t WQ_GMMFR_SPLIT_SIZE = 64L;
constexpr int64_t WQ_GMMFR_SPLIT_FACTOR = 2L;
constexpr int64_t WQ_MAX_NUM_EXPERTS = 1024L;
constexpr int64_t K_ALIGN_SIZE = 32L;
constexpr int64_t N_ALIGN_SIZE = 32L;

static const std::initializer_list<op::DataType> A8W4_X_TYPE_LIST = {op::DataType::DT_FLOAT8_E4M3FN};
static const std::initializer_list<op::DataType> A8W4_W_TYPE_LIST = {op::DataType::DT_FLOAT4_E2M1};
static const std::initializer_list<op::DataType> A8W4_SCALE_TYPE_LIST = {op::DataType::DT_FLOAT8_E8M0};
static const std::initializer_list<op::DataType> A8W4_PERTOKEN_SCALE_TYPE_LIST = {op::DataType::DT_FLOAT8_E8M0};
static const std::initializer_list<op::DataType> A8W4_BIAS_TYPE_LIST = {op::DataType::DT_BF16};
static const std::initializer_list<op::DataType> A8W4_GROUP_LIST_TYPE_LIST = {op::DataType::DT_INT64};
static const std::initializer_list<op::DataType> A8W4_SHARED_INPUT_TYPE_LIST = {op::DataType::DT_BF16};
static const std::initializer_list<op::DataType> A8W4_LOGIT_TYPE_LIST = {op::DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> A8W4_ROW_INDEX_TYPE_LIST = {op::DataType::DT_INT64};
static const std::initializer_list<op::DataType> A8W4_OUT_TYPE_LIST = {op::DataType::DT_FLOAT};

class GroupedMatmulFinalizeRoutingWeightQuant950Checker {
public:
    GroupedMatmulFinalizeRoutingWeightQuant950Checker() = default;

    aclnnStatus CheckParams(GroupedMatmulParams &gmmParams)
    {
        gmmParams_ = gmmParams;
        // 1. 检查参数是否为空指针
        CHECK_RET(CheckNotNull() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR);
        // 2. 检查输入的数据类型是否在支持的数据类型范围之内
        CHECK_RET(CheckDtypeValid(), ACLNN_ERR_PARAM_INVALID);
        // 3. 校验转置属性
        CHECK_RET(CheckTranspose() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
        // 4. 校验输入、输出参数维度
        CHECK_RET(CheckInputOutDims() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
        CHECK_RET(CheckAttrs() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
        // 5. 校验输入、输出shape参数
        CHECK_RET(CheckInputOutShape(), ACLNN_ERR_PARAM_INVALID);
        // 6. 检查数据形状是否支持
        CHECK_RET(CheckFormat(), ACLNN_ERR_PARAM_INVALID);
        return ACLNN_SUCCESS;
    }

    aclnnStatus CheckAttrs()
    {
        const std::string groupListReason = "Attr groupListType must be 0 or 1";
        const std::string offsetNonnegativeReason = "sharedInputOffset must be greater than or equal to 0";
        if (unlikely(gmmParams_.groupListType != 0 && gmmParams_.groupListType != 1)) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ACLNN_NAME, "groupListType", std::to_string(gmmParams_.groupListType),
                                                  groupListReason);
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (unlikely(gmmParams_.shareInputOffset < 0)) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ACLNN_NAME, "sharedInputOffset",
                                                  std::to_string(gmmParams_.shareInputOffset), offsetNonnegativeReason);
            return ACLNN_ERR_PARAM_INVALID;
        }
        int64_t batch = gmmParams_.out->GetViewShape().GetDim(0);
        const std::string offsetUpperReason =
            "sharedInputOffset must be less than or equal to batch(" + std::to_string(batch) + ")";
        if (unlikely(gmmParams_.shareInputOffset > batch)) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ACLNN_NAME, "sharedInputOffset",
                                                  std::to_string(gmmParams_.shareInputOffset), offsetUpperReason);
            return ACLNN_ERR_PARAM_INVALID;
        }
        return ACLNN_SUCCESS;
    }

    aclnnStatus CheckNotNull()
    {
        CHECK_COND(gmmParams_.x1 != nullptr, ACLNN_ERR_PARAM_NULLPTR, "x should not be nullptr.");
        CHECK_COND(gmmParams_.x2 != nullptr, ACLNN_ERR_PARAM_NULLPTR, "weight should not be nullptr.");
        CHECK_COND(gmmParams_.scale != nullptr, ACLNN_ERR_PARAM_NULLPTR, "scale should not be nullptr.");
        CHECK_COND(gmmParams_.groupList != nullptr, ACLNN_ERR_PARAM_NULLPTR, "groupList should not be nullptr.");
        CHECK_COND(gmmParams_.logit != nullptr, ACLNN_ERR_PARAM_NULLPTR, "logit should not be nullptr.");
        CHECK_COND(gmmParams_.rowIndex != nullptr, ACLNN_ERR_PARAM_NULLPTR, "rowIndex should not be nullptr.");
        CHECK_COND(gmmParams_.out != nullptr, ACLNN_ERR_PARAM_NULLPTR, "out should not be nullptr.");
        CHECK_COND(gmmParams_.pertokenScaleOptional != nullptr, ACLNN_ERR_PARAM_NULLPTR,
                   "In A8W4 quant, pertokenScaleOptional should not be nullptr.");
        CHECK_COND(gmmParams_.offset == nullptr, ACLNN_ERR_PARAM_INVALID, "A8W4 quant mode does not support offset.");
        return ACLNN_SUCCESS;
    }

    aclnnStatus CheckTranspose()
    {
        if (unlikely(!(gmmParams_.transposeX1 == false))) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(ACLNN_NAME, "transposeX1", "true",
                                                  "In A8W4 quant, x1 cannot be transposed (transposeX1 must be false)");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (unlikely(!(gmmParams_.transposeX2 == true))) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                ACLNN_NAME, "transposeX2", "false",
                "In A8W4 quant, weight must be transposed (transposeX2 must be true)");
            return ACLNN_ERR_PARAM_INVALID;
        }
        return ACLNN_SUCCESS;
    }

    aclnnStatus CheckInputOutDims()
    {
        auto xDimNumber = gmmParams_.x1->GetViewShape().GetDimNum();
        auto wDimNumber = gmmParams_.x2->GetViewShape().GetDimNum();
        auto wStorageDimNumber = gmmParams_.x2->GetStorageShape().GetDimNum();
        auto wScaleDimNumber = gmmParams_.scale->GetViewShape().GetDimNum();
        auto xScaleDimNumber = gmmParams_.pertokenScaleOptional->GetViewShape().GetDimNum();
        auto grouplistDimNumber = gmmParams_.groupList->GetViewShape().GetDimNum();
        auto logitDimNumber = gmmParams_.logit->GetViewShape().GetDimNum();
        auto rowindexDimNumber = gmmParams_.rowIndex->GetViewShape().GetDimNum();
        auto outDimNumber = gmmParams_.out->GetViewShape().GetDimNum();
        if (unlikely(!(xDimNumber == WQ_DIM_TWO))) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(ACLNN_NAME, "x", std::to_string(xDimNumber),
                                                     "The shape dim of x should be equal to 2");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (unlikely(!(wDimNumber == WQ_DIM_THREE))) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(ACLNN_NAME, "weight", std::to_string(wDimNumber),
                                                     "The shape dim of weight should be equal to 3");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (unlikely(!(wStorageDimNumber == WQ_DIM_FIVE))) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(ACLNN_NAME, "weight(storage)", std::to_string(wStorageDimNumber),
                                                     "The StorageShape dim of weight should be equal to 5");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (unlikely(!(wScaleDimNumber == WQ_DIM_FOUR))) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(ACLNN_NAME, "scale", std::to_string(wScaleDimNumber),
                                                     "The shape dim of scale should be equal to 4");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (unlikely(!(xScaleDimNumber == WQ_DIM_THREE))) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(ACLNN_NAME, "pertokenScale", std::to_string(xScaleDimNumber),
                                                     "The shape dim of pertokenScale should be equal to 3");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (unlikely(!(grouplistDimNumber == WQ_DIM_ONE))) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(ACLNN_NAME, "groupList", std::to_string(grouplistDimNumber),
                                                     "The shape dim of groupList should be equal to 1");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (unlikely(!(logitDimNumber == WQ_DIM_ONE))) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(ACLNN_NAME, "logit", std::to_string(logitDimNumber),
                                                     "The shape dim of logit should be equal to 1");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (unlikely(!(rowindexDimNumber == WQ_DIM_ONE))) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(ACLNN_NAME, "rowIndex", std::to_string(rowindexDimNumber),
                                                     "The shape dim of rowIndex should be equal to 1");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (unlikely(!(outDimNumber == WQ_DIM_TWO))) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(ACLNN_NAME, "out", std::to_string(outDimNumber),
                                                     "The shape dim of out should be equal to 2");
            return ACLNN_ERR_PARAM_INVALID;
        }
        if (gmmParams_.bias != nullptr) {
            auto biasDimNumber = gmmParams_.bias->GetViewShape().GetDimNum();
            if (unlikely(!(biasDimNumber == WQ_DIM_TWO))) {
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(ACLNN_NAME, "bias", std::to_string(biasDimNumber),
                                                         "The shape dim of bias should be equal to 2");
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
        if (gmmParams_.shareInput != nullptr) {
            auto shareInputDimNumber = gmmParams_.shareInput->GetViewShape().GetDimNum();
            if (unlikely(!(shareInputDimNumber == WQ_DIM_TWO))) {
                OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(ACLNN_NAME, "shareInput", std::to_string(shareInputDimNumber),
                                                         "The shape dim of shareInput should be equal to 2");
                return ACLNN_ERR_PARAM_INVALID;
            }
        }
        return ACLNN_SUCCESS;
    }

    bool CheckInputOutShape()
    {
        if (!CheckInputOutShapeConsistency()) {
            return false;
        }
        int64_t m = gmmParams_.x1->GetViewShape().GetDim(WQ_DIM_ZERO);
        int64_t k = gmmParams_.x1->GetViewShape().GetDim(WQ_DIM_ONE);
        // MxA8W4 only supports transposed weight: viewShape=(e, n, k)
        int64_t n = (gmmParams_.x2)->GetViewShape().GetDim(WQ_DIM_ONE);
        int64_t e = (gmmParams_.x2)->GetViewShape().GetDim(0);
        int64_t outputBS = gmmParams_.out->GetViewShape().GetDim(0);
        if (unlikely(k <= 0)) {
            // 保证M/N非0已校验
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(ACLNN_NAME, "k", std::to_string(k), "The k must be positive");
            return false;
        }
        // A8W4约束: k % 32 == 0
        if (unlikely(k % K_ALIGN_SIZE != 0)) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(ACLNN_NAME, "k", std::to_string(k),
                                                  "The k must be a multiple of " + std::to_string(K_ALIGN_SIZE));
            return false;
        }
        // A8W4约束: n % 32 == 0
        if (unlikely(n % N_ALIGN_SIZE != 0)) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(ACLNN_NAME, "n", std::to_string(n),
                                                  "The n must be a multiple of " + std::to_string(N_ALIGN_SIZE));
            return false;
        }
        if (!CheckRequiredShapes(m, k, n, e, outputBS)) {
            return false;
        }
        if (!CheckOptionalShapes(e, n)) {
            return false;
        }
        return true;
    }

    bool CheckRequiredShapes(int64_t m, int64_t k, int64_t n, int64_t e, int64_t outputBS)
    {
        op::Shape xExpectShape = {m, k};
        // MxA8W4 only supports transposed weight: viewShape=(e, n, k)
        op::Shape weightExpectShape = {e, n, k};
        // MxA8W4 scale shape: (e, n, ceil(k/64), 2)
        op::Shape weightScaleExpectShape =
            op::Shape{e, n, Ops::Base::CeilDiv(k, WQ_GMMFR_SPLIT_SIZE), WQ_GMMFR_SPLIT_FACTOR};
        op::Shape xScaleExpectShape = op::Shape{m, Ops::Base::CeilDiv(k, WQ_GMMFR_SPLIT_SIZE), WQ_GMMFR_SPLIT_FACTOR};
        op::Shape grouplistExpectShape = {e};
        op::Shape logitExpectShape = {m};
        op::Shape rowindexExpectShape = {m};
        op::Shape outputExpectShape = {outputBS, n};
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(gmmParams_.x1, xExpectShape, return false);
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(gmmParams_.x2, weightExpectShape, return false);
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(gmmParams_.scale, weightScaleExpectShape, return false);
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(gmmParams_.pertokenScaleOptional, xScaleExpectShape, return false);
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(gmmParams_.groupList, grouplistExpectShape, return false);
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(gmmParams_.logit, logitExpectShape, return false);
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(gmmParams_.rowIndex, rowindexExpectShape, return false);
        OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(gmmParams_.out, outputExpectShape, return false);
        return true;
    }

    bool CheckOptionalShapes(int64_t e, int64_t n)
    {
        if (gmmParams_.bias != nullptr) {
            op::Shape biasExpectShape = {e, n};
            OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(gmmParams_.bias, biasExpectShape, return false);
        }
        if (gmmParams_.shareInput != nullptr) {
            int64_t bsdp = gmmParams_.shareInput->GetViewShape().GetDim(0);
            if (unlikely(bsdp > gmmParams_.out->GetViewShape().GetDim(0))) {
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(ACLNN_NAME, "shareInput batch", std::to_string(bsdp),
                                                      "The batch of shareInput must be <= the batch of out " +
                                                          std::to_string(gmmParams_.out->GetViewShape().GetDim(0)));
                return false;
            }
            if (unlikely(gmmParams_.shareInputOffset > gmmParams_.out->GetViewShape().GetDim(0) - bsdp)) {
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    ACLNN_NAME, "sharedInputOffset + shareInput batch",
                    std::to_string(gmmParams_.shareInputOffset) + " + " + std::to_string(bsdp),
                    "sharedInputOffset + batch of shareInput must be <= batch of out (" +
                        std::to_string(gmmParams_.out->GetViewShape().GetDim(0)) + ")");
                return false;
            }
            op::Shape shareInputExpectShape = {bsdp, n};
            OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(gmmParams_.shareInput, shareInputExpectShape, return false);
        }
        return true;
    }

    bool CheckInputOutShapeConsistency()
    {
        int64_t k = gmmParams_.x1->GetViewShape().GetDim(WQ_DIM_ONE);
        // MxA8W4 only supports transposed weight: viewShape=(e, n, k), so k is at dim 2
        int64_t kInWeight = gmmParams_.x2->GetViewShape().GetDim(WQ_DIM_TWO);
        int64_t e = (gmmParams_.x2)->GetViewShape().GetDim(0);
        if (unlikely(kInWeight != k)) {
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(ACLNN_NAME, "x, weight",
                                                   std::to_string(k) + ", " + std::to_string(kInWeight),
                                                   "The axis (k) of x must be equal to the axis (k) of weight");
            return false;
        }
        int64_t groupListLen = gmmParams_.groupList->GetViewShape().GetDim(WQ_DIM_ZERO);
        if (unlikely(groupListLen != e)) {
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                ACLNN_NAME, "groupList, weight", std::to_string(groupListLen) + ", " + std::to_string(e),
                "Length of groupList must be equal to the number of experts in weight");
            return false;
        }
        if (unlikely(e > WQ_MAX_NUM_EXPERTS)) {
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                ACLNN_NAME, "e", std::to_string(e),
                "In A8W4 quant, e must be less than or equal to " + std::to_string(WQ_MAX_NUM_EXPERTS));
            return false;
        }
        return true;
    }

    bool CheckDtypeValid()
    {
        OP_CHECK_DTYPE_NOT_SUPPORT_WITH_REASON(ACLNN_NAME, gmmParams_.x1, A8W4_X_TYPE_LIST, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT_WITH_REASON(ACLNN_NAME, gmmParams_.x2, A8W4_W_TYPE_LIST, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT_WITH_REASON(ACLNN_NAME, gmmParams_.scale, A8W4_SCALE_TYPE_LIST, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT_WITH_REASON(ACLNN_NAME, gmmParams_.rowIndex, A8W4_ROW_INDEX_TYPE_LIST, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT_WITH_REASON(ACLNN_NAME, gmmParams_.pertokenScaleOptional,
                                               A8W4_PERTOKEN_SCALE_TYPE_LIST, return false);
        if (gmmParams_.bias != nullptr) {
            OP_CHECK_DTYPE_NOT_SUPPORT_WITH_REASON(ACLNN_NAME, gmmParams_.bias, A8W4_BIAS_TYPE_LIST, return false);
        }
        OP_CHECK_DTYPE_NOT_SUPPORT_WITH_REASON(ACLNN_NAME, gmmParams_.groupList, A8W4_GROUP_LIST_TYPE_LIST,
                                               return false);
        if (gmmParams_.shareInput != nullptr) {
            OP_CHECK_DTYPE_NOT_SUPPORT_WITH_REASON(ACLNN_NAME, gmmParams_.shareInput, A8W4_SHARED_INPUT_TYPE_LIST,
                                                   return false);
        }
        OP_CHECK_DTYPE_NOT_SUPPORT_WITH_REASON(ACLNN_NAME, gmmParams_.logit, A8W4_LOGIT_TYPE_LIST, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT_WITH_REASON(ACLNN_NAME, gmmParams_.out, A8W4_OUT_TYPE_LIST, return false);
        return true;
    }

    bool CheckFormat()
    {
        if (!CheckXAndWeightFormat()) {
            return false;
        }
        if (!CheckOtherTensorFormats()) {
            return false;
        }
        return true;
    }

    bool CheckXAndWeightFormat()
    {
        if (unlikely(op::IsPrivateFormat(gmmParams_.x1->GetStorageFormat()))) {
            OP_LOGE_FOR_INVALID_FORMAT(ACLNN_NAME, "x", op::ToString(gmmParams_.x1->GetStorageFormat()).GetString(),
                                       "ND");
            return false;
        }
        // A8W4场景：权重为NZ格式
        if (unlikely(gmmParams_.x2->GetStorageFormat() != Format::FORMAT_FRACTAL_NZ &&
                     gmmParams_.x2->GetStorageFormat() != Format::FORMAT_FRACTAL_NZ_C0_32)) {
            OP_LOGE_FOR_INVALID_FORMAT(ACLNN_NAME, "weight",
                                       op::ToString(gmmParams_.x2->GetStorageFormat()).GetString(), "NZ(C0_32)");
            return false;
        }
        return true;
    }

    bool CheckOtherTensorFormats()
    {
        if (unlikely(op::IsPrivateFormat(gmmParams_.pertokenScaleOptional->GetStorageFormat()))) {
            OP_LOGE_FOR_INVALID_FORMAT(ACLNN_NAME, "pertokenScaleOptional",
                                       op::ToString(gmmParams_.pertokenScaleOptional->GetStorageFormat()).GetString(),
                                       "ND");
            return false;
        }
        if (unlikely(op::IsPrivateFormat(gmmParams_.scale->GetStorageFormat()))) {
            OP_LOGE_FOR_INVALID_FORMAT(ACLNN_NAME, "scale",
                                       op::ToString(gmmParams_.scale->GetStorageFormat()).GetString(), "ND");
            return false;
        }
        if (unlikely(op::IsPrivateFormat(gmmParams_.groupList->GetStorageFormat()))) {
            OP_LOGE_FOR_INVALID_FORMAT(ACLNN_NAME, "groupList",
                                       op::ToString(gmmParams_.groupList->GetStorageFormat()).GetString(), "ND");
            return false;
        }
        if (unlikely(op::IsPrivateFormat(gmmParams_.logit->GetStorageFormat()))) {
            OP_LOGE_FOR_INVALID_FORMAT(ACLNN_NAME, "logit",
                                       op::ToString(gmmParams_.logit->GetStorageFormat()).GetString(), "ND");
            return false;
        }
        if (unlikely(op::IsPrivateFormat(gmmParams_.rowIndex->GetStorageFormat()))) {
            OP_LOGE_FOR_INVALID_FORMAT(ACLNN_NAME, "rowIndex",
                                       op::ToString(gmmParams_.rowIndex->GetStorageFormat()).GetString(), "ND");
            return false;
        }
        if (unlikely(op::IsPrivateFormat(gmmParams_.out->GetStorageFormat()))) {
            OP_LOGE_FOR_INVALID_FORMAT(ACLNN_NAME, "out", op::ToString(gmmParams_.out->GetStorageFormat()).GetString(),
                                       "ND");
            return false;
        }
        if (gmmParams_.bias != nullptr && unlikely(op::IsPrivateFormat(gmmParams_.bias->GetStorageFormat()))) {
            OP_LOGE_FOR_INVALID_FORMAT(ACLNN_NAME, "bias",
                                       op::ToString(gmmParams_.bias->GetStorageFormat()).GetString(), "ND");
            return false;
        }
        if (gmmParams_.shareInput != nullptr &&
            unlikely(op::IsPrivateFormat(gmmParams_.shareInput->GetStorageFormat()))) {
            OP_LOGE_FOR_INVALID_FORMAT(ACLNN_NAME, "shareInput",
                                       op::ToString(gmmParams_.shareInput->GetStorageFormat()).GetString(), "ND");
            return false;
        }
        return true;
    }

private:
    GroupedMatmulParams gmmParams_;
};
} // namespace GmmFinalizeRouting
#endif
