/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MOE_FINALIZE_ROUTING_COMMON_H
#define MOE_FINALIZE_ROUTING_COMMON_H

#include "log/log.h"
#include <string>
#include "aclnn_kernels/common/op_error_check.h"
#include "op_host/util/op_const_def.h"
#include "external/aclnn_kernels/aclnn_platform.h"

using namespace op;

namespace MoeFinalizeRoutingCheck {

static inline bool Is310P()
{
    return GetCurrentPlatformInfo().GetCurNpuArch() == Ops::Base::DAV_2002;
}

static inline bool IsCommonValidationChip()
{
    return Ops::Transformer::AclnnUtil::IsRegbase() || GetCurrentPlatformInfo().GetCurNpuArch() == Ops::Base::DAV_2201;
}

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_X = {op::DataType::DT_FLOAT16,
                                                                         op::DataType::DT_BF16, op::DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_X_310P = {op::DataType::DT_FLOAT16,
                                                                              op::DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_MX4_AUX = {op::DataType::DT_BF16};
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_MX4_SCALE = {op::DataType::DT_FLOAT8_E8M0};
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_ROW_IDX = {op::DataType::DT_INT32};

static inline bool CheckNotNull(const aclTensor* expandedX, const aclTensor* expandedRowIdx, const aclTensor* out)
{
    OP_CHECK_NULL(expandedX, return false);
    OP_CHECK_NULL(expandedRowIdx, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static inline bool CheckDtypeValid(const aclTensor* expandedX, const aclTensor* expandedRowIdx,
                                   const aclTensor* x1Optional, const aclTensor* x2Optional,
                                   const aclTensor* biasOptional, const aclTensor* scalesOptional,
                                   const aclTensor* expertIdxOptional, const aclTensor* out, bool isMx4 = false)
{
    const auto& auxTypes = isMx4 ? DTYPE_SUPPORT_LIST_MX4_AUX : DTYPE_SUPPORT_LIST_X;
    const auto& scaleTypes = isMx4 ? DTYPE_SUPPORT_LIST_MX4_SCALE : DTYPE_SUPPORT_LIST_X;
    const aclTensor* reference = isMx4 ? out : expandedX;
    if (expandedX != nullptr && !isMx4 && expandedX->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(expandedX, DTYPE_SUPPORT_LIST_X, return false);
    }
    if (expandedRowIdx != nullptr && (isMx4 || expandedRowIdx->GetViewShape().GetShapeSize() != 0)) {
        OP_CHECK_DTYPE_NOT_SUPPORT(expandedRowIdx, DTYPE_SUPPORT_LIST_ROW_IDX, return false);
    }
    if (x1Optional != nullptr && (isMx4 || x1Optional->GetViewShape().GetShapeSize() != 0)) {
        OP_CHECK_DTYPE_NOT_SUPPORT(x1Optional, auxTypes, return false);
        OP_CHECK_DTYPE_NOT_SAME(reference, x1Optional, return false);
    }
    if (x2Optional != nullptr && (isMx4 || x2Optional->GetViewShape().GetShapeSize() != 0)) {
        OP_CHECK_DTYPE_NOT_SUPPORT(x2Optional, auxTypes, return false);
        OP_CHECK_DTYPE_NOT_SAME(reference, x2Optional, return false);
    }
    if (biasOptional != nullptr && (isMx4 || biasOptional->GetViewShape().GetShapeSize() != 0)) {
        OP_CHECK_DTYPE_NOT_SUPPORT(biasOptional, auxTypes, return false);
        OP_CHECK_DTYPE_NOT_SAME(reference, biasOptional, return false);
    }
    if (scalesOptional != nullptr && (isMx4 || scalesOptional->GetViewShape().GetShapeSize() != 0)) {
        OP_CHECK_DTYPE_NOT_SUPPORT(scalesOptional, scaleTypes, return false);
    }
    if (expertIdxOptional != nullptr && (isMx4 || expertIdxOptional->GetViewShape().GetShapeSize() != 0)) {
        OP_CHECK_DTYPE_NOT_SUPPORT(expertIdxOptional, DTYPE_SUPPORT_LIST_ROW_IDX, return false);
    }
    if (out != nullptr && (isMx4 || out->GetViewShape().GetShapeSize() != 0)) {
        OP_CHECK_DTYPE_NOT_SUPPORT(out, auxTypes, return false);
        OP_CHECK_DTYPE_NOT_SAME(reference, out, return false);
    }
    return true;
}

static inline bool CheckDtypeValid310P(const aclTensor* expandedX, const aclTensor* expandedRowIdx,
                                       const aclTensor* x1Optional, const aclTensor* x2Optional,
                                       const aclTensor* biasOptional, const aclTensor* scalesOptional,
                                       const aclTensor* expertIdxOptional, const aclTensor* out)
{
    if (expandedX != nullptr && expandedX->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(expandedX, DTYPE_SUPPORT_LIST_X_310P, return false);
    }
    if (expandedRowIdx != nullptr && expandedRowIdx->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(expandedRowIdx, DTYPE_SUPPORT_LIST_ROW_IDX, return false);
    }
    if (x1Optional != nullptr && x1Optional->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(x1Optional, DTYPE_SUPPORT_LIST_X_310P, return false);
        OP_CHECK_DTYPE_NOT_SAME(expandedX, x1Optional, return false);
    }
    if (x2Optional != nullptr && x2Optional->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(x2Optional, DTYPE_SUPPORT_LIST_X_310P, return false);
        OP_CHECK_DTYPE_NOT_SAME(expandedX, x2Optional, return false);
    }
    if (biasOptional != nullptr && biasOptional->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(biasOptional, DTYPE_SUPPORT_LIST_X_310P, return false);
        OP_CHECK_DTYPE_NOT_SAME(expandedX, biasOptional, return false);
    }
    if (scalesOptional != nullptr && scalesOptional->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(scalesOptional, DTYPE_SUPPORT_LIST_X_310P, return false);
    }
    if (expertIdxOptional != nullptr && expertIdxOptional->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(expertIdxOptional, DTYPE_SUPPORT_LIST_ROW_IDX, return false);
    }
    if (out != nullptr && out->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(out, DTYPE_SUPPORT_LIST_X_310P, return false);
        OP_CHECK_DTYPE_NOT_SAME(expandedX, out, return false);
    }
    return true;
}

static inline bool CheckConstExpertDtype(const aclTensor* expandedX, const aclTensor* xOptional,
                                         const aclTensor* alpha1Optional, const aclTensor* alpha2Optional,
                                         const aclTensor* vOptional, bool isMx4 = false)
{
    const auto& types = isMx4 ? DTYPE_SUPPORT_LIST_MX4_AUX : DTYPE_SUPPORT_LIST_X;
    const aclTensor* extras[] = {xOptional, alpha1Optional, alpha2Optional, vOptional};
    for (const auto* tensor : extras) {
        if (tensor != nullptr && (isMx4 || tensor->GetViewShape().GetShapeSize() != 0)) {
            OP_CHECK_DTYPE_NOT_SUPPORT(tensor, types, return false);
            if (!isMx4) {
                OP_CHECK_DTYPE_NOT_SAME(expandedX, tensor, return false);
            }
        }
    }
    return true;
}

static inline bool IsExpandedX3D(int64_t dropPadMode)
{
    return dropPadMode != 0 && dropPadMode != 2;
}

static inline bool CheckFormatValid(const aclTensor* expandedX, const aclTensor* expandedRowIdx,
                                    const aclTensor* x1Optional, const aclTensor* x2Optional,
                                    const aclTensor* biasOptional, const aclTensor* scalesOptional,
                                    const aclTensor* expertIdxOptional, const aclTensor* xOptional,
                                    const aclTensor* alpha1Optional, const aclTensor* alpha2Optional,
                                    const aclTensor* vOptional, const aclTensor* out, int64_t dropPadMode,
                                    bool isMx4 = false)
{
    bool expandedXIs3D = !isMx4 && IsExpandedX3D(dropPadMode);
    if (expandedX != nullptr && (isMx4 || expandedX->GetViewShape().GetShapeSize() != 0)) {
        op::Format format = isMx4 ? expandedX->GetViewFormat() : expandedX->GetStorageFormat();
        // aclnn converts the format of 3D tensors to NCL
        bool isSupportedFormat = format == op::Format::FORMAT_ND || (expandedXIs3D && format == op::Format::FORMAT_NCL);
        if (!isSupportedFormat) {
            OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(
                "MoeFinalizeRoutingV2", "expanded_x",
                std::string(op::ToString(format).GetString()) + ", " + std::string(expandedXIs3D ? " and NCL" : ""),
                "expandedX format is " + std::string(op::ToString(format).GetString()) + ", only ND" +
                    std::string(expandedXIs3D ? " and NCL" : "") + " is supported.");
            return false;
        }
    }
    std::vector<const aclTensor*> tensors = {expandedRowIdx, x1Optional,        x2Optional, biasOptional,
                                             scalesOptional, expertIdxOptional, xOptional,  alpha1Optional,
                                             alpha2Optional, vOptional,         out};
    std::vector<std::string> names = {"expandedRowIdx", "x1",     "x2", "bias", "scales", "expertIdx", "x",
                                      "alpha1",         "alpha2", "v",  "out"};
    for (size_t i = 0; i < tensors.size(); ++i) {
        if (tensors[i] != nullptr && (isMx4 || tensors[i]->GetViewShape().GetShapeSize() != 0)) {
            const auto format = isMx4 ? tensors[i]->GetViewFormat() : tensors[i]->GetStorageFormat();
            if (format != op::Format::FORMAT_ND) {
                OP_LOGE_FOR_INVALID_FORMAT_WITH_REASON(
                    "MoeFinalizeRoutingV2", names[i],
                    std::string(names[i].c_str()) + ", " +
                        std::string(op::ToString(tensors[i]->GetStorageFormat()).GetString()),
                    "Tensor " + std::string(names[i].c_str()) + " format is " +
                        std::string(op::ToString(tensors[i]->GetStorageFormat()).GetString()) +
                        ", only ND format is supported.");
                return false;
            }
        }
    }
    return true;
}

static inline bool CheckShapeValid(const aclTensor* expandedX, const aclTensor* expandedRowIdx,
                                   const aclTensor* x1Optional, const aclTensor* x2Optional,
                                   const aclTensor* biasOptional, const aclTensor* scalesOptional,
                                   const aclTensor* expertIdxOptional, const aclTensor* xOptional,
                                   const aclTensor* alpha1Optional, const aclTensor* alpha2Optional,
                                   const aclTensor* vOptional, const aclTensor* out, int64_t dropPadMode,
                                   int64_t mx4K = 0, int64_t constantRows = 0)
{
    size_t expandedXDimNum = expandedX->GetViewShape().GetDimNum();
    if (IsExpandedX3D(dropPadMode)) {
        if (expandedXDimNum != 3) {
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("MoeFinalizeRoutingV2", "expanded_x, drop_pad_mode",
                                                     "expandedX should be 3D when dropPadMode is " +
                                                         std::to_string(dropPadMode) + ", but got " +
                                                         std::to_string(expandedXDimNum) + "D.");
            return false;
        }
    } else {
        if (expandedXDimNum != 2) {
            OP_LOGE_FOR_INVALID_ARGUMENT_WITH_REASON("MoeFinalizeRoutingV2", "expanded_x, drop_pad_mode",
                                                     "expandedX should be 2D when dropPadMode is " +
                                                         std::to_string(dropPadMode) + ", but got " +
                                                         std::to_string(expandedXDimNum) + "D.");
            return false;
        }
    }
    if (expandedRowIdx->GetViewShape().GetDimNum() != 1) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("MoeFinalizeRoutingV2", "expanded_row_idx",
                                                 std::to_string(expandedRowIdx->GetViewShape().GetDimNum()),
                                                 "expandedRowIdx should be 1D, but got " +
                                                     std::to_string(expandedRowIdx->GetViewShape().GetDimNum()) + "D.");
        return false;
    }
    const bool isMx4 = mx4K > 0;
    const int64_t rows = isMx4 ? expandedRowIdx->GetViewShape().GetDim(0) / mx4K : 0;
    const int64_t width = expandedX->GetViewShape().GetDim(expandedXDimNum - 1);
    struct OptionalShape {
        const aclTensor* tensor;
        const char* name;
        int64_t rows;
        int64_t cols;
    };
    const OptionalShape optional2D[] = {{x1Optional, "x1", rows, width},
                                        {x2Optional, "x2", rows, width},
                                        {biasOptional, "bias", -1, width},
                                        {scalesOptional, "scales", rows, mx4K},
                                        {expertIdxOptional, "expertIdx", rows, mx4K},
                                        {xOptional, "x", rows, width},
                                        {alpha1Optional, "alpha1", constantRows, width},
                                        {alpha2Optional, "alpha2", constantRows, width},
                                        {vOptional, "v", constantRows, width}};
    for (const auto& item : optional2D) {
        if (item.tensor == nullptr || (!isMx4 && item.tensor->GetViewShape().GetShapeSize() == 0)) {
            continue;
        }
        const auto& shape = item.tensor->GetViewShape();
        if (shape.GetDimNum() != 2) {
            OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON("MoeFinalizeRoutingV2", item.name,
                                                     std::to_string(shape.GetDimNum()),
                                                     "Tensor " + std::string(item.name) + " should be 2D, but got " +
                                                         std::to_string(shape.GetDimNum()) + "D.");
            return false;
        }
        if (isMx4 && ((item.rows >= 0 && shape.GetDim(0) != item.rows) || shape.GetDim(1) != item.cols)) {
            return false;
        }
    }
    if (out->GetViewShape().GetDimNum() != 2) {
        OP_LOGE_FOR_INVALID_SHAPEDIM_WITH_REASON(
            "MoeFinalizeRoutingV2", "out", std::to_string(out->GetViewShape().GetDimNum()),
            "out should be 2D, but got " + std::to_string(out->GetViewShape().GetDimNum()) + "D.");
        return false;
    }
    return !isMx4 || out->GetViewShape() == op::Shape{rows, width};
}

static inline aclnnStatus CheckParams(const aclTensor* expandedX, const aclTensor* expandedRowIdx,
                                      const aclTensor* x1Optional, const aclTensor* x2Optional,
                                      const aclTensor* biasOptional, const aclTensor* scalesOptional,
                                      const aclTensor* expertIdxOptional, const aclTensor* xOptional,
                                      const aclTensor* alpha1Optional, const aclTensor* alpha2Optional,
                                      const aclTensor* vOptional, const aclTensor* out, int64_t dropPadMode)
{
    CHECK_RET(CheckNotNull(expandedX, expandedRowIdx, out), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtypeValid(expandedX, expandedRowIdx, x1Optional, x2Optional, biasOptional, scalesOptional,
                              expertIdxOptional, out),
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(
        CheckFormatValid(expandedX, expandedRowIdx, x1Optional, x2Optional, biasOptional, scalesOptional,
                         expertIdxOptional, xOptional, alpha1Optional, alpha2Optional, vOptional, out, dropPadMode),
        ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(
        CheckShapeValid(expandedX, expandedRowIdx, x1Optional, x2Optional, biasOptional, scalesOptional,
                        expertIdxOptional, xOptional, alpha1Optional, alpha2Optional, vOptional, out, dropPadMode),
        ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static inline aclnnStatus CheckParams310P(const aclTensor* expandedX, const aclTensor* expandedRowIdx,
                                          const aclTensor* x1Optional, const aclTensor* x2Optional,
                                          const aclTensor* biasOptional, const aclTensor* scalesOptional,
                                          const aclTensor* expertIdxOptional, const aclTensor* out, int64_t dropPadMode)
{
    CHECK_RET(CheckNotNull(expandedX, expandedRowIdx, out), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtypeValid310P(expandedX, expandedRowIdx, x1Optional, x2Optional, biasOptional, scalesOptional,
                                  expertIdxOptional, out),
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckFormatValid(expandedX, expandedRowIdx, x1Optional, x2Optional, biasOptional, scalesOptional,
                               expertIdxOptional, nullptr, nullptr, nullptr, nullptr, out, dropPadMode),
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShapeValid(expandedX, expandedRowIdx, x1Optional, x2Optional, biasOptional, scalesOptional,
                              expertIdxOptional, nullptr, nullptr, nullptr, nullptr, out, dropPadMode),
              ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

} // namespace MoeFinalizeRoutingCheck

#endif // MOE_FINALIZE_ROUTING_COMMON_H
