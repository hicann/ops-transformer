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

#include "aclnn_kernels/common/op_error_check.h"
#include "external/aclnn_kernels/aclnn_platform.h"

using namespace op;

namespace MoeFinalizeRoutingCheck {

static inline bool Is310P()
{
    return GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND310P;
}

static inline bool IsCommonValidationChip()
{
    return Ops::Transformer::AclnnUtil::IsRegbase() ||
           GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910B ||
           GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND910_93;
}

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_X = {op::DataType::DT_FLOAT16,
                                                                         op::DataType::DT_BF16, op::DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_X_310P = {op::DataType::DT_FLOAT16,
                                                                              op::DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_LIST_ROW_IDX = {op::DataType::DT_INT32};

static inline bool CheckNotNull(const aclTensor *expandedX, const aclTensor *expandedRowIdx, const aclTensor *out)
{
    OP_CHECK_NULL(expandedX, return false);
    OP_CHECK_NULL(expandedRowIdx, return false);
    OP_CHECK_NULL(out, return false);
    return true;
}

static inline bool CheckDtypeValid(const aclTensor *expandedX, const aclTensor *expandedRowIdx,
                                   const aclTensor *x1Optional, const aclTensor *x2Optional,
                                   const aclTensor *biasOptional, const aclTensor *scalesOptional,
                                   const aclTensor *expertIdxOptional, const aclTensor *out)
{
    if (expandedX != nullptr && expandedX->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(expandedX, DTYPE_SUPPORT_LIST_X, return false);
    }
    if (expandedRowIdx != nullptr && expandedRowIdx->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(expandedRowIdx, DTYPE_SUPPORT_LIST_ROW_IDX, return false);
    }
    if (x1Optional != nullptr && x1Optional->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(x1Optional, DTYPE_SUPPORT_LIST_X, return false);
        OP_CHECK_DTYPE_NOT_SAME(expandedX, x1Optional, return false);
    }
    if (x2Optional != nullptr && x2Optional->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(x2Optional, DTYPE_SUPPORT_LIST_X, return false);
        OP_CHECK_DTYPE_NOT_SAME(expandedX, x2Optional, return false);
    }
    if (biasOptional != nullptr && biasOptional->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(biasOptional, DTYPE_SUPPORT_LIST_X, return false);
        OP_CHECK_DTYPE_NOT_SAME(expandedX, biasOptional, return false);
    }
    if (scalesOptional != nullptr && scalesOptional->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(scalesOptional, DTYPE_SUPPORT_LIST_X, return false);
    }
    if (expertIdxOptional != nullptr && expertIdxOptional->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(expertIdxOptional, DTYPE_SUPPORT_LIST_ROW_IDX, return false);
    }
    if (out != nullptr && out->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(out, DTYPE_SUPPORT_LIST_X, return false);
        OP_CHECK_DTYPE_NOT_SAME(expandedX, out, return false);
    }
    return true;
}

static inline bool CheckDtypeValid310P(const aclTensor *expandedX, const aclTensor *expandedRowIdx,
                                       const aclTensor *x1Optional, const aclTensor *x2Optional,
                                       const aclTensor *biasOptional, const aclTensor *scalesOptional,
                                       const aclTensor *expertIdxOptional, const aclTensor *out)
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

static inline bool CheckConstExpertDtype(const aclTensor *expandedX, const aclTensor *xOptional,
                                         const aclTensor *alpha1Optional, const aclTensor *alpha2Optional,
                                         const aclTensor *vOptional)
{
    const aclTensor *extras[] = {xOptional, alpha1Optional, alpha2Optional, vOptional};
    for (const auto *tensor : extras) {
        if (tensor != nullptr && tensor->GetViewShape().GetShapeSize() != 0) {
            OP_CHECK_DTYPE_NOT_SUPPORT(tensor, DTYPE_SUPPORT_LIST_X, return false);
            OP_CHECK_DTYPE_NOT_SAME(expandedX, tensor, return false);
        }
    }
    return true;
}

static inline bool IsExpandedX3D(int64_t dropPadMode)
{
    return dropPadMode != 0 && dropPadMode != 2;
}

static inline bool CheckFormatValid(const aclTensor *expandedX, const aclTensor *expandedRowIdx,
                                    const aclTensor *x1Optional, const aclTensor *x2Optional,
                                    const aclTensor *biasOptional, const aclTensor *scalesOptional,
                                    const aclTensor *expertIdxOptional, const aclTensor *xOptional,
                                    const aclTensor *alpha1Optional, const aclTensor *alpha2Optional,
                                    const aclTensor *vOptional, const aclTensor *out, int64_t dropPadMode)
{
    bool expandedXIs3D = IsExpandedX3D(dropPadMode);
    if (expandedX != nullptr && expandedX->GetViewShape().GetShapeSize() != 0) {
        op::Format format = expandedX->GetStorageFormat();
        // aclnn converts the format of 3D tensors to NCL
        bool isSupportedFormat = format == op::Format::FORMAT_ND || (expandedXIs3D && format == op::Format::FORMAT_NCL);
        if (!isSupportedFormat) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "expandedX format is %s, only ND%s is supported.",
                    op::ToString(format).GetString(), expandedXIs3D ? " and NCL" : "");
            return false;
        }
    }
    std::vector<const aclTensor *> tensors = {expandedRowIdx, x1Optional,        x2Optional, biasOptional,
                                              scalesOptional, expertIdxOptional, xOptional,  alpha1Optional,
                                              alpha2Optional, vOptional,         out};
    std::vector<std::string> names = {"expandedRowIdx", "x1",     "x2", "bias", "scales", "expertIdx", "x",
                                      "alpha1",         "alpha2", "v",  "out"};
    for (size_t i = 0; i < tensors.size(); ++i) {
        if (tensors[i] != nullptr && tensors[i]->GetViewShape().GetShapeSize() != 0) {
            if (tensors[i]->GetStorageFormat() != op::Format::FORMAT_ND) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Tensor %s format is %s, only ND format is supported.",
                        names[i].c_str(), op::ToString(tensors[i]->GetStorageFormat()).GetString());
                return false;
            }
        }
    }
    return true;
}

static inline bool CheckShapeValid(const aclTensor *expandedX, const aclTensor *expandedRowIdx,
                                   const aclTensor *x1Optional, const aclTensor *x2Optional,
                                   const aclTensor *biasOptional, const aclTensor *scalesOptional,
                                   const aclTensor *expertIdxOptional, const aclTensor *xOptional,
                                   const aclTensor *alpha1Optional, const aclTensor *alpha2Optional,
                                   const aclTensor *vOptional, const aclTensor *out, int64_t dropPadMode)
{
    size_t expandedXDimNum = expandedX->GetViewShape().GetDimNum();
    if (IsExpandedX3D(dropPadMode)) {
        if (expandedXDimNum != 3) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "expandedX should be 3D when dropPadMode is %ld, but got %zuD.",
                    dropPadMode, expandedXDimNum);
            return false;
        }
    } else {
        if (expandedXDimNum != 2) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "expandedX should be 2D when dropPadMode is %ld, but got %zuD.",
                    dropPadMode, expandedXDimNum);
            return false;
        }
    }
    if (expandedRowIdx->GetViewShape().GetDimNum() != 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "expandedRowIdx should be 1D, but got %zuD.",
                expandedRowIdx->GetViewShape().GetDimNum());
        return false;
    }
    std::vector<std::pair<const aclTensor *, std::string>> optional2D = {{x1Optional, "x1"},
                                                                         {x2Optional, "x2"},
                                                                         {biasOptional, "bias"},
                                                                         {scalesOptional, "scales"},
                                                                         {expertIdxOptional, "expertIdx"},
                                                                         {xOptional, "x"},
                                                                         {alpha1Optional, "alpha1"},
                                                                         {alpha2Optional, "alpha2"},
                                                                         {vOptional, "v"}};
    for (const auto &item : optional2D) {
        if (item.first != nullptr && item.first->GetViewShape().GetShapeSize() != 0) {
            if (item.first->GetViewShape().GetDimNum() != 2) {
                OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Tensor %s should be 2D, but got %zuD.", item.second.c_str(),
                        item.first->GetViewShape().GetDimNum());
                return false;
            }
        }
    }
    if (out->GetViewShape().GetDimNum() != 2) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "out should be 2D, but got %zuD.", out->GetViewShape().GetDimNum());
        return false;
    }
    return true;
}

static inline aclnnStatus CheckParams(const aclTensor *expandedX, const aclTensor *expandedRowIdx,
                                      const aclTensor *x1Optional, const aclTensor *x2Optional,
                                      const aclTensor *biasOptional, const aclTensor *scalesOptional,
                                      const aclTensor *expertIdxOptional, const aclTensor *xOptional,
                                      const aclTensor *alpha1Optional, const aclTensor *alpha2Optional,
                                      const aclTensor *vOptional, const aclTensor *out, int64_t dropPadMode)
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

static inline aclnnStatus CheckParams310P(const aclTensor *expandedX, const aclTensor *expandedRowIdx,
                                          const aclTensor *x1Optional, const aclTensor *x2Optional,
                                          const aclTensor *biasOptional, const aclTensor *scalesOptional,
                                          const aclTensor *expertIdxOptional, const aclTensor *out, int64_t dropPadMode)
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
