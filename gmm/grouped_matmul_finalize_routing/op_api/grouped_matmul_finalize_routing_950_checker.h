/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_API_INC_GROUPED_MATMUL_FINALIZE_ROUTING_950_CHECKER_H
#define OP_API_INC_GROUPED_MATMUL_FINALIZE_ROUTING_950_CHECKER_H
#include "opdev/format_utils.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "quant_grouped_matmul_finalize_routing_util.h"
#include "../../grouped_matmul/op_api/grouped_matmul_util.h"
#include "util/math_util.h"
#include "log/log.h"

#include <sstream>

namespace GmmFinalizeRouting {

inline std::string ShapeToStringWithoutBracket(const op::Shape &shape)
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

inline std::string ViewShapeToString(const aclTensor *tensor)
{
    return ShapeToStringWithoutBracket(tensor->GetViewShape());
}

#define GMMFR_CHECK_REPORT(cond, retExpr, reportExpr) \
    do { \
        if (!(cond)) { \
            reportExpr; \
            retExpr; \
        } \
    } while (0)

#define GMMFR_CHECK_SHAPE(tensor, paramName, expectedShape, retExpr) \
    do { \
        if ((tensor)->GetViewShape() != (expectedShape)) { \
            OP_LOGE_FOR_INVALID_SHAPE(GMMFR_ACLNN_OP_NAME, paramName, \
                                      op::ToString((tensor)->GetViewShape()).GetString(), \
                                      op::ToString(expectedShape).GetString()); \
            retExpr; \
        } \
    } while (0)

#define GMMFR_CHECK_DTYPE(tensor, paramName, supportList, retExpr) \
    do { \
        if (!CheckType((tensor)->GetDataType(), supportList)) { \
            OP_LOGE_FOR_INVALID_DTYPE(GMMFR_ACLNN_OP_NAME, paramName, \
                                      op::ToString((tensor)->GetDataType()).GetString(), \
                                      op::ToString(supportList).GetString()); \
            retExpr; \
        } \
    } while (0)

constexpr size_t ZERO_DIM = 0UL;
constexpr size_t ONE_DIM = 1UL;
constexpr size_t TWO_DIM = 2UL;
constexpr size_t THREE_DIM = 3UL;
constexpr size_t FOUR_DIM = 4UL;
constexpr int64_t GMMFR_SPLIT_SIZE = 64L;
constexpr int64_t GMMFR_SPLIT_FACTOR = 2L;
constexpr int64_t MOD2 = 2L;
constexpr int64_t MAX_NUM_EXPERTS = 1024L;
constexpr const char *GMMFR_ACLNN_OP_NAME = "aclnnGroupedMatmulFinalizeRoutingGetWorkspaceSize";

const std::initializer_list<DataType> X_WEIGHT_TYPE_SUPPORT_LIST_MX = {
    op::DataType::DT_FLOAT8_E4M3FN, op::DataType::DT_FLOAT8_E5M2, op::DataType::DT_FLOAT4_E2M1};
const std::initializer_list<DataType> X_WEIGHT_TYPE_SUPPORT_LIST_FP4 = {op::DataType::DT_FLOAT4_E2M1};
const std::initializer_list<DataType> X_WEIGHT_TYPE_SUPPORT_LIST_FP8 = {op::DataType::DT_FLOAT4_E2M1};
static const std::initializer_list<op::DataType> SCALE_TYPE_SUPPORT_LIST_MX = {op::DataType::DT_FLOAT8_E8M0};
static const std::initializer_list<op::DataType> ROW_INDEX_TYPE_SUPPORT_LIST_MX = {op::DataType::DT_INT64};
static const std::initializer_list<op::DataType> BIAS_TYPE_SUPPORT_LIST_MX = {op::DataType::DT_BF16};
static const std::initializer_list<op::DataType> PERTOKEN_SCALE_TYPE_SUPPORT_LIST_MX = {op::DataType::DT_FLOAT8_E8M0};
static const std::initializer_list<op::DataType> GROUP_LIST_TYPE_SUPPORT_LIST = {op::DataType::DT_INT64};
static const std::initializer_list<op::DataType> SHARED_INPUT_TYPE_SUPPORT_LIST = {op::DataType::DT_BF16};
static const std::initializer_list<op::DataType> LOGIT_TYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> OUT_TYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT};

const std::initializer_list<DataType> X_WEIGHT_TYPE_SUPPORT_LIST_PERTOKEN = {
    DataType::DT_INT8, DataType::DT_FLOAT8_E4M3FN, DataType::DT_HIFLOAT8};
static const std::initializer_list<op::DataType> PERTOKEN_SCALE_TYPE_SUPPORT_LIST_PERTOKEN = {op::DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> BIAS_TYPE_SUPPORT_LIST_PERTOKEN = {op::DataType::DT_BF16};
static const std::initializer_list<op::DataType> SCALE_TYPE_SUPPORT_LIST_PERTOKEN = {op::DataType::DT_FLOAT,
                                                                                     op::DataType::DT_BF16};
static const std::initializer_list<op::DataType> ROW_INDEX_TYPE_SUPPORT_LIST_PERTOKEN_INT8 = {op::DataType::DT_INT64,
                                                                                              op::DataType::DT_INT32};
static const std::initializer_list<op::DataType> ROW_INDEX_TYPE_SUPPORT_LIST_PERTOKEN_FP8HIFLOAT8 = {
    op::DataType::DT_INT64};
enum class QuantMode {
    PERTOKEN = 0, // pertoken 量化
    MX = 2        // MX量化
};

class AclnnGroupedMatmulFinalizeRoutingDAV3510Checker {
public:
    explicit AclnnGroupedMatmulFinalizeRoutingDAV3510Checker() {};
    ~AclnnGroupedMatmulFinalizeRoutingDAV3510Checker() {};
    aclnnStatus CheckParams(GroupedMatmulParams &gmmParams)
    {
        gmmParams_ = gmmParams;
        // 0. 进入判断逻辑之前先判断是哪种量化
        GMMFR_CHECK_REPORT(gmmParams_.scale != nullptr, return ACLNN_ERR_PARAM_NULLPTR,
                           OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(GMMFR_ACLNN_OP_NAME, "scale", "nullptr",
                                                                 "the value of scale cannot be nullptr"));
        DataType scaleDtype = gmmParams_.scale->GetDataType();
        if (CheckType(scaleDtype, SCALE_TYPE_SUPPORT_LIST_MX)) {
            quantMode_ = QuantMode::MX;
        } else if (CheckType(scaleDtype, SCALE_TYPE_SUPPORT_LIST_PERTOKEN)) {
            quantMode_ = QuantMode::PERTOKEN;
        } else {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                GMMFR_ACLNN_OP_NAME, "scale", op::ToString(scaleDtype).GetString(),
                "the dtype of scale must be within the range {FLOAT8_E8M0, FLOAT32, BFLOAT16}");
            return ACLNN_ERR_PARAM_INVALID;
        }
        // 1. 检查参数是否为空指针
        CHECK_RET(CheckNotNull() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_NULLPTR);
        // 2. 检查输入的数据类型是否在支持的数据类型范围之内
        CHECK_RET(CheckDtypeValid(), ACLNN_ERR_PARAM_INVALID);
        // 3. 校验输入、输出参数维度
        CHECK_RET(CheckInputOutDims() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
        CHECK_RET(CheckAttrs() == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
        // 4. 校验输入、输出shape参数
        CHECK_RET(CheckInputOutShape(), ACLNN_ERR_PARAM_INVALID);
        // 5. 校验输入、输出shape参数针对MXFP4
        if (CheckType(gmmParams_.x1->GetDataType(), X_WEIGHT_TYPE_SUPPORT_LIST_FP4)) {
            CHECK_RET(CheckInputOutShapeForMXFP4(), ACLNN_ERR_PARAM_INVALID);
        }
        // 6. 检查数据形状是否支持
        CHECK_RET(CheckFormat(), ACLNN_ERR_PARAM_INVALID);
        return ACLNN_SUCCESS;
    }

    aclnnStatus CheckAttrs()
    {
        const std::string offsetNonnegativeReason = "sharedInputOffset must be greater than or equal to 0";
        GMMFR_CHECK_REPORT(gmmParams_.groupListType == 0 || gmmParams_.groupListType == 1,
                           return ACLNN_ERR_PARAM_INVALID,
                           OP_LOGE_FOR_INVALID_VALUE(GMMFR_ACLNN_OP_NAME, "groupListType",
                                                     std::to_string(gmmParams_.groupListType), "0 or 1"));
        GMMFR_CHECK_REPORT(gmmParams_.shareInputOffset >= 0, return ACLNN_ERR_PARAM_INVALID,
                           OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(GMMFR_ACLNN_OP_NAME, "sharedInputOffset",
                                                                 std::to_string(gmmParams_.shareInputOffset),
                                                                 offsetNonnegativeReason));
        int64_t batch = gmmParams_.out->GetViewShape().GetDim(ZERO_DIM);
        const std::string offsetUpperReason =
            "sharedInputOffset must be less than or equal to batch(" + std::to_string(batch) + ")";
        GMMFR_CHECK_REPORT(
            gmmParams_.shareInputOffset <= batch, return ACLNN_ERR_PARAM_INVALID,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(GMMFR_ACLNN_OP_NAME, "sharedInputOffset",
                                                  std::to_string(gmmParams_.shareInputOffset), offsetUpperReason));
        GMMFR_CHECK_REPORT(!gmmParams_.transposeX1, return ACLNN_ERR_PARAM_INVALID,
                           OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(GMMFR_ACLNN_OP_NAME, "transposeX1", "true",
                                                                 "x cannot be transposed on Ascend 950"));
        return ACLNN_SUCCESS;
    }

    aclnnStatus CheckNotNull()
    {
        GMMFR_CHECK_REPORT(gmmParams_.x1 != nullptr, return ACLNN_ERR_PARAM_NULLPTR,
                           OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(GMMFR_ACLNN_OP_NAME, "x", "nullptr",
                                                                 "the value of x cannot be nullptr"));
        GMMFR_CHECK_REPORT(gmmParams_.x2 != nullptr, return ACLNN_ERR_PARAM_NULLPTR,
                           OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(GMMFR_ACLNN_OP_NAME, "weight", "nullptr",
                                                                 "the value of weight cannot be nullptr"));
        GMMFR_CHECK_REPORT(gmmParams_.scale != nullptr, return ACLNN_ERR_PARAM_NULLPTR,
                           OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(GMMFR_ACLNN_OP_NAME, "scale", "nullptr",
                                                                 "the value of scale cannot be nullptr"));
        GMMFR_CHECK_REPORT(gmmParams_.groupList != nullptr, return ACLNN_ERR_PARAM_NULLPTR,
                           OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(GMMFR_ACLNN_OP_NAME, "groupList", "nullptr",
                                                                 "the value of groupList cannot be nullptr"));
        GMMFR_CHECK_REPORT(gmmParams_.logit != nullptr, return ACLNN_ERR_PARAM_NULLPTR,
                           OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(GMMFR_ACLNN_OP_NAME, "logit", "nullptr",
                                                                 "the value of logit cannot be nullptr"));
        GMMFR_CHECK_REPORT(gmmParams_.rowIndex != nullptr, return ACLNN_ERR_PARAM_NULLPTR,
                           OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(GMMFR_ACLNN_OP_NAME, "rowIndex", "nullptr",
                                                                 "the value of rowIndex cannot be nullptr"));
        GMMFR_CHECK_REPORT(gmmParams_.out != nullptr, return ACLNN_ERR_PARAM_NULLPTR,
                           OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(GMMFR_ACLNN_OP_NAME, "y", "nullptr",
                                                                 "the value of y cannot be nullptr"));
        GMMFR_CHECK_REPORT(gmmParams_.offset == nullptr, return ACLNN_ERR_PARAM_INVALID,
                           OP_LOGE_FOR_INVALID_VALUE(GMMFR_ACLNN_OP_NAME, "offset", "nonnull", "nullptr"));
        if (quantMode_ == QuantMode::MX) {
            GMMFR_CHECK_REPORT(
                gmmParams_.pertokenScaleOptional != nullptr, return ACLNN_ERR_PARAM_NULLPTR,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(GMMFR_ACLNN_OP_NAME, "perTokenScale", "nullptr",
                                                      "in MX quant, the value of perTokenScale cannot be nullptr"));
        }
        return ACLNN_SUCCESS;
    }

    aclnnStatus CheckInputOutDims()
    {
        auto xDimNumber = gmmParams_.x1->GetViewShape().GetDimNum();
        auto wDimNumber = gmmParams_.x2->GetViewShape().GetDimNum();
        auto wScaleDimNumber = gmmParams_.scale->GetViewShape().GetDimNum();
        auto grouplistDimNumber = gmmParams_.groupList->GetViewShape().GetDimNum();
        auto logitDimNumber = gmmParams_.logit->GetViewShape().GetDimNum();
        auto rowindexDimNumber = gmmParams_.rowIndex->GetViewShape().GetDimNum();
        auto outDimNumber = gmmParams_.out->GetViewShape().GetDimNum();
        size_t xscaleExpectDim = quantMode_ == QuantMode::MX ? THREE_DIM : ONE_DIM;
        size_t weightscaleExpectDim = quantMode_ == QuantMode::MX ? FOUR_DIM : THREE_DIM;
        GMMFR_CHECK_REPORT(xDimNumber == TWO_DIM, return ACLNN_ERR_PARAM_INVALID,
                           OP_LOGE_FOR_INVALID_SHAPEDIM(GMMFR_ACLNN_OP_NAME, "x", std::to_string(xDimNumber),
                                                        std::to_string(TWO_DIM)));
        GMMFR_CHECK_REPORT(wDimNumber == THREE_DIM, return ACLNN_ERR_PARAM_INVALID,
                           OP_LOGE_FOR_INVALID_SHAPEDIM(GMMFR_ACLNN_OP_NAME, "weight", std::to_string(wDimNumber),
                                                        std::to_string(THREE_DIM)));
        GMMFR_CHECK_REPORT(wScaleDimNumber == weightscaleExpectDim, return ACLNN_ERR_PARAM_INVALID,
                           OP_LOGE_FOR_INVALID_SHAPEDIM(GMMFR_ACLNN_OP_NAME, "scale", std::to_string(wScaleDimNumber),
                                                        std::to_string(weightscaleExpectDim)));
        GMMFR_CHECK_REPORT(grouplistDimNumber == ONE_DIM, return ACLNN_ERR_PARAM_INVALID,
                           OP_LOGE_FOR_INVALID_SHAPEDIM(GMMFR_ACLNN_OP_NAME, "groupList",
                                                        std::to_string(grouplistDimNumber), std::to_string(ONE_DIM)));
        GMMFR_CHECK_REPORT(logitDimNumber == ONE_DIM, return ACLNN_ERR_PARAM_INVALID,
                           OP_LOGE_FOR_INVALID_SHAPEDIM(GMMFR_ACLNN_OP_NAME, "logit", std::to_string(logitDimNumber),
                                                        std::to_string(ONE_DIM)));
        GMMFR_CHECK_REPORT(rowindexDimNumber == ONE_DIM, return ACLNN_ERR_PARAM_INVALID,
                           OP_LOGE_FOR_INVALID_SHAPEDIM(GMMFR_ACLNN_OP_NAME, "rowIndex",
                                                        std::to_string(rowindexDimNumber), std::to_string(ONE_DIM)));
        GMMFR_CHECK_REPORT(outDimNumber == TWO_DIM, return ACLNN_ERR_PARAM_INVALID,
                           OP_LOGE_FOR_INVALID_SHAPEDIM(GMMFR_ACLNN_OP_NAME, "y", std::to_string(outDimNumber),
                                                        std::to_string(TWO_DIM)));
        if (gmmParams_.pertokenScaleOptional != nullptr) {
            auto xScaleDimNumber = gmmParams_.pertokenScaleOptional->GetViewShape().GetDimNum();
            GMMFR_CHECK_REPORT(
                xScaleDimNumber == xscaleExpectDim, return ACLNN_ERR_PARAM_INVALID,
                OP_LOGE_FOR_INVALID_SHAPEDIM(GMMFR_ACLNN_OP_NAME, "perTokenScale", std::to_string(xScaleDimNumber),
                                             std::to_string(xscaleExpectDim)));
        }
        if (gmmParams_.bias != nullptr) {
            auto baisDimNumber = gmmParams_.bias->GetViewShape().GetDimNum();
            GMMFR_CHECK_REPORT(baisDimNumber == TWO_DIM, return ACLNN_ERR_PARAM_INVALID,
                               OP_LOGE_FOR_INVALID_SHAPEDIM(GMMFR_ACLNN_OP_NAME, "bias", std::to_string(baisDimNumber),
                                                            std::to_string(TWO_DIM)));
        }
        if (gmmParams_.shareInput != nullptr) {
            auto shareInputDimNumber = gmmParams_.shareInput->GetViewShape().GetDimNum();
            GMMFR_CHECK_REPORT(
                shareInputDimNumber == TWO_DIM, return ACLNN_ERR_PARAM_INVALID,
                OP_LOGE_FOR_INVALID_SHAPEDIM(GMMFR_ACLNN_OP_NAME, "shareInput", std::to_string(shareInputDimNumber),
                                             std::to_string(TWO_DIM)));
        }
        return ACLNN_SUCCESS;
    }

    bool CheckInputOutShape()
    {
        if (CheckInputOutShapeConsistency() == false) {
            return false;
        }
        int64_t m = gmmParams_.x1->GetViewShape().GetDim(ZERO_DIM); // 从x的第0维获取m
        int64_t k = gmmParams_.x1->GetViewShape().GetDim(ONE_DIM);  // 从x的第1维获取k
        // 转置情况下从weight的第1维获取n，非转置情况下从weight的第2维获取n
        int64_t n = gmmParams_.transposeX2 ? (gmmParams_.x2)->GetViewShape().GetDim(ONE_DIM) :
                                             (gmmParams_.x2)->GetViewShape().GetDim(TWO_DIM);
        int64_t e = (gmmParams_.x2)->GetViewShape().GetDim(0); // 从weight的第0维获取e
        int64_t outputBS = gmmParams_.out->GetViewShape().GetDim(0);
        if (k <= 0) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(GMMFR_ACLNN_OP_NAME, "x", ViewShapeToString(gmmParams_.x1),
                                                  "when the M or N value is not 0 [M=" + std::to_string(m) + ", N=" +
                                                      std::to_string(n) + "], axis K of x must be a positive number");
            return false;
        }
        if (!CheckRequiredShapes(m, k, n, e, outputBS)) {
            return false;
        }
        if (!CheckOptionalShapes(m, k, e, n)) {
            return false;
        }
        return true;
    }

    bool CheckRequiredShapes(int64_t m, int64_t k, int64_t n, int64_t e, int64_t outputBS)
    {
        op::Shape xExpectShape = {m, k};
        op::Shape weightExpectShape = {e, k, n};
        op::Shape weightScaleExpectShape =
            quantMode_ == QuantMode::MX ? op::Shape{e, Ops::Base::CeilDiv(k, GMMFR_SPLIT_SIZE), n, GMMFR_SPLIT_FACTOR} :
                                          op::Shape{e, 1, n};
        op::Shape weightTransExpectShape = {e, n, k};
        op::Shape weightScaleTransExpectShape =
            quantMode_ == QuantMode::MX ? op::Shape{e, n, Ops::Base::CeilDiv(k, GMMFR_SPLIT_SIZE), GMMFR_SPLIT_FACTOR} :
                                          op::Shape{e, 1, n};
        op::Shape grouplistExpectShape = {e};
        op::Shape logitExpectShape = {m};
        op::Shape rowindexExpectShape = {m};
        op::Shape outputExpectShape = {outputBS, n};
        GMMFR_CHECK_SHAPE(gmmParams_.x1, "x", xExpectShape, return false);
        if (gmmParams_.transposeX2) {
            GMMFR_CHECK_SHAPE(gmmParams_.scale, "scale", weightScaleTransExpectShape, return false);
            GMMFR_CHECK_SHAPE(gmmParams_.x2, "weight", weightTransExpectShape, return false);
        } else {
            GMMFR_CHECK_SHAPE(gmmParams_.scale, "scale", weightScaleExpectShape, return false);
            GMMFR_CHECK_SHAPE(gmmParams_.x2, "weight", weightExpectShape, return false);
        }
        GMMFR_CHECK_SHAPE(gmmParams_.groupList, "groupList", grouplistExpectShape, return false);
        GMMFR_CHECK_SHAPE(gmmParams_.logit, "logit", logitExpectShape, return false);
        GMMFR_CHECK_SHAPE(gmmParams_.rowIndex, "rowIndex", rowindexExpectShape, return false);
        GMMFR_CHECK_SHAPE(gmmParams_.out, "y", outputExpectShape, return false);
        return true;
    }

    bool CheckOptionalShapes(int64_t m, int64_t k, int64_t e, int64_t n)
    {
        if (gmmParams_.pertokenScaleOptional != nullptr) {
            op::Shape xScaleExpectShape =
                quantMode_ == QuantMode::MX ?
                    op::Shape{m, Ops::Base::CeilDiv(k, GMMFR_SPLIT_SIZE), GMMFR_SPLIT_FACTOR} :
                    op::Shape{m};
            GMMFR_CHECK_SHAPE(gmmParams_.pertokenScaleOptional, "perTokenScale", xScaleExpectShape, return false);
        }
        if (gmmParams_.bias != nullptr) {
            op::Shape biasExpectShape = {e, n};
            GMMFR_CHECK_SHAPE(gmmParams_.bias, "bias", biasExpectShape, return false);
        }
        if (gmmParams_.shareInput != nullptr) {
            int64_t bsdp = gmmParams_.shareInput->GetViewShape().GetDim(0);
            int64_t batch = gmmParams_.out->GetViewShape().GetDim(0);
            GMMFR_CHECK_REPORT(
                bsdp <= batch, return false,
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    GMMFR_ACLNN_OP_NAME, "shareInput", std::to_string(bsdp),
                    "shared_input_len must be less than or equal to batch(" + std::to_string(batch) + ")"));
            GMMFR_CHECK_REPORT(
                gmmParams_.shareInputOffset <= batch - bsdp, return false,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
                    GMMFR_ACLNN_OP_NAME, "sharedInputOffset", std::to_string(gmmParams_.shareInputOffset),
                    "sharedInputOffset plus sharedInputLen must be less than or equal to batch(" +
                        std::to_string(batch) + ")"));
            op::Shape shareInputExpectShape = {bsdp, n};
            GMMFR_CHECK_SHAPE(gmmParams_.shareInput, "shareInput", shareInputExpectShape, return false);
        }
        return true;
    }

    bool CheckInputOutShapeConsistency()
    {
        int64_t k = gmmParams_.x1->GetViewShape().GetDim(ONE_DIM); // 从x的第1维获取k
        int64_t kInWeight = gmmParams_.transposeX2 ? (gmmParams_.x2)->GetViewShape().GetDim(TWO_DIM) :
                                                     (gmmParams_.x2)->GetViewShape().GetDim(ONE_DIM);
        int64_t e = (gmmParams_.x2)->GetViewShape().GetDim(0); // 从weight的第0维获取e
        if (kInWeight != k) {
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(
                GMMFR_ACLNN_OP_NAME, "x and weight",
                "x=" + ViewShapeToString(gmmParams_.x1) + ", weight=" + ViewShapeToString(gmmParams_.x2),
                "axis K of x must be equal to axis K of weight");
            return false;
        }
        // groupList的长度应等于weight的专家数
        int64_t groupListLen = gmmParams_.groupList->GetViewShape().GetDim(ZERO_DIM);
        if (groupListLen != e) {
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                GMMFR_ACLNN_OP_NAME, "groupList", ViewShapeToString(gmmParams_.groupList),
                "the total number of elements of groupList must be equal to the number of experts in weight [" +
                    std::to_string(e) + "]");
            return false;
        }
        if (e > MAX_NUM_EXPERTS) {
            const char *quantModeStr = (quantMode_ == QuantMode::MX) ? "MX quant" : "pertoken quant";
            OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                GMMFR_ACLNN_OP_NAME, "weight", ViewShapeToString(gmmParams_.x2),
                "in " + std::string(quantModeStr) + ", e must be less than or equal to 1024");
            return false;
        }
        return true;
    }

    bool CheckInputOutShapeForMXFP4()
    {
        int64_t k = gmmParams_.x1->GetViewShape().GetDim(ONE_DIM); // 从x的第1维获取k
        // 转置情况下从weight的第1维获取n，非转置情况下从weight的第2维获取n
        int64_t n = gmmParams_.transposeX2 ? (gmmParams_.x2)->GetViewShape().GetDim(ONE_DIM) :
                                             (gmmParams_.x2)->GetViewShape().GetDim(TWO_DIM);
        DataType xDtype = gmmParams_.x1->GetDataType();
        DataType weightDtype = gmmParams_.x2->GetDataType();
        if (xDtype == DataType::DT_FLOAT4_E2M1 && weightDtype == DataType::DT_FLOAT4_E2M1) {
            if (!(k % MOD2 == 0)) {
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(GMMFR_ACLNN_OP_NAME, "x", ViewShapeToString(gmmParams_.x1),
                                                       "in MXFP4, k must be divisible by 2");
                return false;
            }
            if (gmmParams_.transposeX2 == false && (n % MOD2 != 0)) {
                OP_LOGE_FOR_INVALID_SHAPE_WITH_REASON(
                    GMMFR_ACLNN_OP_NAME, "weight", ViewShapeToString(gmmParams_.x2),
                    "in MXFP4 and when x2 is not transposed, axis N of weight must be an even number");
                return false;
            }
            if (k == MOD2) {
                OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(GMMFR_ACLNN_OP_NAME, "x", ViewShapeToString(gmmParams_.x1),
                                                       "in MXFP4, k cannot be 2");
                return false;
            }
        }
        return true;
    }

    bool CheckDtypeValid()
    {
        DataType scaleDtype = gmmParams_.scale->GetDataType();
        if (quantMode_ == QuantMode::MX) {
            if (CheckDtypeValidForMX() == false) {
                return false;
            }
        } else if (quantMode_ == QuantMode::PERTOKEN) {
            if (CheckDtypeValidForPertoken() == false) {
                return false;
            }
        } else {
            OP_LOGE_FOR_INVALID_DTYPE_WITH_REASON(
                GMMFR_ACLNN_OP_NAME, "scale", op::ToString(scaleDtype).GetString(),
                "the dtype of scale must be within the range {FLOAT8_E8M0, FLOAT32, BFLOAT16}");
            return false;
        }
        return true;
    }

    bool CheckDtypeValidForMX()
    {
        GMMFR_CHECK_DTYPE(gmmParams_.x1, "x", X_WEIGHT_TYPE_SUPPORT_LIST_MX, return false);
        GMMFR_CHECK_DTYPE(gmmParams_.x2, "weight", X_WEIGHT_TYPE_SUPPORT_LIST_MX, return false);
        GMMFR_CHECK_DTYPE(gmmParams_.scale, "scale", SCALE_TYPE_SUPPORT_LIST_MX, return false);
        GMMFR_CHECK_DTYPE(gmmParams_.rowIndex, "rowIndex", ROW_INDEX_TYPE_SUPPORT_LIST_MX, return false);
        GMMFR_CHECK_DTYPE(gmmParams_.pertokenScaleOptional, "perTokenScale", PERTOKEN_SCALE_TYPE_SUPPORT_LIST_MX,
                          return false);
        if (gmmParams_.bias != nullptr) {
            GMMFR_CHECK_DTYPE(gmmParams_.bias, "bias", BIAS_TYPE_SUPPORT_LIST_MX, return false);
        }
        GMMFR_CHECK_DTYPE(gmmParams_.groupList, "groupList", GROUP_LIST_TYPE_SUPPORT_LIST, return false);
        if (gmmParams_.shareInput != nullptr) {
            GMMFR_CHECK_DTYPE(gmmParams_.shareInput, "shareInput", SHARED_INPUT_TYPE_SUPPORT_LIST, return false);
        }
        GMMFR_CHECK_DTYPE(gmmParams_.logit, "logit", LOGIT_TYPE_SUPPORT_LIST, return false);
        GMMFR_CHECK_DTYPE(gmmParams_.out, "y", OUT_TYPE_SUPPORT_LIST, return false);
        if ((CheckType(gmmParams_.x1->GetDataType(), X_WEIGHT_TYPE_SUPPORT_LIST_FP4) !=
             CheckType(gmmParams_.x2->GetDataType(), X_WEIGHT_TYPE_SUPPORT_LIST_FP4)) ||
            (CheckType(gmmParams_.x1->GetDataType(), X_WEIGHT_TYPE_SUPPORT_LIST_FP8) !=
             CheckType(gmmParams_.x2->GetDataType(), X_WEIGHT_TYPE_SUPPORT_LIST_FP8))) {
            OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                GMMFR_ACLNN_OP_NAME, "x and weight",
                "x=" + std::string(op::ToString(gmmParams_.x1->GetDataType()).GetString()) +
                    ", weight=" + op::ToString(gmmParams_.x2->GetDataType()).GetString(),
                "the dtypes of x and weight must be within the range {both MXFP4 or both MXFP8}");
            return false;
        }
        return true;
    }

    bool CheckDtypeValidForPertoken()
    {
        GMMFR_CHECK_DTYPE(gmmParams_.x1, "x", X_WEIGHT_TYPE_SUPPORT_LIST_PERTOKEN, return false);
        GMMFR_CHECK_DTYPE(gmmParams_.x2, "weight", X_WEIGHT_TYPE_SUPPORT_LIST_PERTOKEN, return false);
        GMMFR_CHECK_DTYPE(gmmParams_.scale, "scale", SCALE_TYPE_SUPPORT_LIST_PERTOKEN, return false);
        if (gmmParams_.x1->GetDataType() == DataType::DT_INT8) {
            GMMFR_CHECK_DTYPE(gmmParams_.rowIndex, "rowIndex", ROW_INDEX_TYPE_SUPPORT_LIST_PERTOKEN_INT8, return false);
        } else {
            GMMFR_CHECK_DTYPE(gmmParams_.rowIndex, "rowIndex", ROW_INDEX_TYPE_SUPPORT_LIST_PERTOKEN_FP8HIFLOAT8,
                              return false);
        }
        if (gmmParams_.pertokenScaleOptional != nullptr) {
            GMMFR_CHECK_DTYPE(gmmParams_.pertokenScaleOptional, "perTokenScale",
                              PERTOKEN_SCALE_TYPE_SUPPORT_LIST_PERTOKEN, return false);
        }
        if (gmmParams_.bias != nullptr) {
            GMMFR_CHECK_DTYPE(gmmParams_.bias, "bias", BIAS_TYPE_SUPPORT_LIST_PERTOKEN, return false);
        }
        GMMFR_CHECK_DTYPE(gmmParams_.groupList, "groupList", GROUP_LIST_TYPE_SUPPORT_LIST, return false);
        if (gmmParams_.shareInput != nullptr) {
            GMMFR_CHECK_DTYPE(gmmParams_.shareInput, "shareInput", SHARED_INPUT_TYPE_SUPPORT_LIST, return false);
        }
        GMMFR_CHECK_DTYPE(gmmParams_.logit, "logit", LOGIT_TYPE_SUPPORT_LIST, return false);
        GMMFR_CHECK_DTYPE(gmmParams_.out, "y", OUT_TYPE_SUPPORT_LIST, return false);
        if (gmmParams_.x1->GetDataType() != gmmParams_.x2->GetDataType()) {
            bool xIsFP8 = CheckType(gmmParams_.x1->GetDataType(), X_WEIGHT_TYPE_SUPPORT_LIST_FP8);
            bool wIsFP8 = CheckType(gmmParams_.x2->GetDataType(), X_WEIGHT_TYPE_SUPPORT_LIST_FP8);
            if (!xIsFP8 || !wIsFP8) {
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                    GMMFR_ACLNN_OP_NAME, "x and weight",
                    "x=" + std::string(op::ToString(gmmParams_.x1->GetDataType()).GetString()) +
                        ", weight=" + op::ToString(gmmParams_.x2->GetDataType()).GetString(),
                    "the dtypes of x and weight must be the same");
                return false;
            }
        }
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
        if (op::IsPrivateFormat(gmmParams_.x1->GetStorageFormat()) ||
            (gmmParams_.pertokenScaleOptional != nullptr &&
             op::IsPrivateFormat(gmmParams_.pertokenScaleOptional->GetStorageFormat()))) {
            std::string perTokenScaleFormat =
                gmmParams_.pertokenScaleOptional == nullptr ?
                    "nullptr" :
                    op::ToString(gmmParams_.pertokenScaleOptional->GetStorageFormat()).GetString();
            OP_LOGE_FOR_INVALID_FORMATS_WITH_REASON(
                GMMFR_ACLNN_OP_NAME, "x and perTokenScale",
                "x=" + std::string(op::ToString(gmmParams_.x1->GetStorageFormat()).GetString()) +
                    ", perTokenScale=" + perTokenScaleFormat,
                "the formats of x and perTokenScale must be ND");
            return false;
        }
        if (quantMode_ == QuantMode::MX) {
            if (op::IsPrivateFormat(gmmParams_.x2->GetStorageFormat())) {
                OP_LOGE_FOR_INVALID_FORMAT(GMMFR_ACLNN_OP_NAME, "weight",
                                           op::ToString(gmmParams_.x2->GetStorageFormat()).GetString(), "ND");
                return false;
            }
        } else {
            if (gmmParams_.x2->GetStorageFormat() != Format::FORMAT_FRACTAL_NZ) {
                OP_LOGE_FOR_INVALID_FORMAT(GMMFR_ACLNN_OP_NAME, "weight",
                                           op::ToString(gmmParams_.x2->GetStorageFormat()).GetString(), "FRACTAL_NZ");
                return false;
            }
        }
        return true;
    }

    bool CheckOtherTensorFormats()
    {
        if (op::IsPrivateFormat(gmmParams_.scale->GetStorageFormat())) {
            OP_LOGE_FOR_INVALID_FORMAT(GMMFR_ACLNN_OP_NAME, "scale",
                                       op::ToString(gmmParams_.scale->GetStorageFormat()).GetString(), "ND");
            return false;
        }
        if (op::IsPrivateFormat(gmmParams_.groupList->GetStorageFormat())) {
            OP_LOGE_FOR_INVALID_FORMAT(GMMFR_ACLNN_OP_NAME, "groupList",
                                       op::ToString(gmmParams_.groupList->GetStorageFormat()).GetString(), "ND");
            return false;
        }
        if (op::IsPrivateFormat(gmmParams_.logit->GetStorageFormat())) {
            OP_LOGE_FOR_INVALID_FORMAT(GMMFR_ACLNN_OP_NAME, "logit",
                                       op::ToString(gmmParams_.logit->GetStorageFormat()).GetString(), "ND");
            return false;
        }
        if (op::IsPrivateFormat(gmmParams_.rowIndex->GetStorageFormat())) {
            OP_LOGE_FOR_INVALID_FORMAT(GMMFR_ACLNN_OP_NAME, "rowIndex",
                                       op::ToString(gmmParams_.rowIndex->GetStorageFormat()).GetString(), "ND");
            return false;
        }
        if (op::IsPrivateFormat(gmmParams_.out->GetStorageFormat())) {
            OP_LOGE_FOR_INVALID_FORMAT(GMMFR_ACLNN_OP_NAME, "y",
                                       op::ToString(gmmParams_.out->GetStorageFormat()).GetString(), "ND");
            return false;
        }
        if (gmmParams_.bias != nullptr && op::IsPrivateFormat(gmmParams_.bias->GetStorageFormat())) {
            OP_LOGE_FOR_INVALID_FORMAT(GMMFR_ACLNN_OP_NAME, "bias",
                                       op::ToString(gmmParams_.bias->GetStorageFormat()).GetString(), "ND");
            return false;
        }
        if (gmmParams_.shareInput != nullptr && op::IsPrivateFormat(gmmParams_.shareInput->GetStorageFormat())) {
            OP_LOGE_FOR_INVALID_FORMAT(GMMFR_ACLNN_OP_NAME, "shareInput",
                                       op::ToString(gmmParams_.shareInput->GetStorageFormat()).GetString(), "ND");
            return false;
        }
        return true;
    }

private:
    GroupedMatmulParams gmmParams_;
    QuantMode quantMode_;
};
#undef GMMFR_CHECK_DTYPE
#undef GMMFR_CHECK_SHAPE
#undef GMMFR_CHECK_REPORT
} // namespace GmmFinalizeRouting
#endif
