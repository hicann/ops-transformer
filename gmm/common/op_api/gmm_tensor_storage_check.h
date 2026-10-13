/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GMM_TENSOR_STORAGE_CHECK_H
#define GMM_TENSOR_STORAGE_CHECK_H

#include <limits>
#include <string>

#include "log/log.h"
#include "opdev/common_types.h"
#include "opdev/format_utils.h"
#include "opdev/shape_utils.h"

namespace gmm {
inline bool ReportInvalidTensorStorage(const aclTensor* tensor, const char* opName, const char* name,
                                       const char* reason)
{
    const std::string layout = std::string("viewShape=") + op::ToString(tensor->GetViewShape()).GetString() +
                               ", strides=" + op::ToString(tensor->GetViewStrides()).GetString() +
                               ", offset=" + std::to_string(tensor->GetViewOffset()) +
                               ", storageShape=" + op::ToString(tensor->GetStorageShape()).GetString();
    OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opName, name, layout, reason);
    return false;
}

// Check the caller's descriptor before any unpacking, CreateView, Contiguous or kernel preparation.
// Use the caller's declared element counts before reinterpreting packed weights as four-bit tensors.
// The packed FP4 NZ layout uses bytes in its innermost storage dimension and
// 4-bit elements in its view; other format-specific checks belong to each operator.
// This validates the declared storage only; aclTensor does not expose the caller's allocation size.
inline bool CheckTensorStorageBoundsWithViewShape(const aclTensor* tensor, const op::Shape& viewShape,
                                                  const char* opName, const char* name)
{
    // Required/optional null semantics belong to each operator's existing parameter checks.
    if (tensor == nullptr) {
        return true;
    }
    const auto& strides = tensor->GetViewStrides();
    const auto& storageShape = tensor->GetStorageShape();
    const int64_t offset = tensor->GetViewOffset();
    if (viewShape.GetDimNum() != strides.size()) {
        return ReportInvalidTensorStorage(tensor, opName, name, "view shape and strides must have the same rank");
    }
    if (offset < 0) {
        return ReportInvalidTensorStorage(tensor, opName, name, "offset must be nonnegative");
    }
    bool emptyView = false;
    for (size_t i = 0; i < viewShape.GetDimNum(); ++i) {
        if (viewShape.GetDim(i) < 0 || strides[i] < 0) {
            return ReportInvalidTensorStorage(tensor, opName, name, "view dimensions and strides must be nonnegative");
        }
        emptyView = emptyView || viewShape.GetDim(i) == 0;
    }
    bool emptyStorage = false;
    for (size_t i = 0; i < storageShape.GetDimNum(); ++i) {
        if (storageShape.GetDim(i) < 0) {
            return ReportInvalidTensorStorage(tensor, opName, name, "storage dimensions must be nonnegative");
        }
        emptyStorage = emptyStorage || storageShape.GetDim(i) == 0;
    }
    // Validate every dimension even for empty tensors, but do not invent an access for an empty view.
    if (emptyView) {
        return true;
    }
    if (emptyStorage) {
        return ReportInvalidTensorStorage(tensor, opName, name, "a nonempty view cannot use empty storage");
    }
    int64_t storageElements = 1;
    for (size_t i = 0; i < storageShape.GetDimNum(); ++i) {
        const int64_t dim = storageShape.GetDim(i);
        // Negative dimensions were rejected above, and zero dimensions return through
        // the empty-storage check before this loop, so dim is strictly positive here.
        if (storageElements > std::numeric_limits<int64_t>::max() / dim) {
            return ReportInvalidTensorStorage(tensor, opName, name, "storage element count overflows int64");
        }
        storageElements *= dim;
    }
    // MXA8W4 FRACTAL_NZ with C0=32 stores two FP4 values per storage-shape entry.
    // MXFP4 with C0=64 and INT4 layouts use logical element counts as before.
    const auto dtype = tensor->GetDataType();
    const auto format = ge::GetPrimaryFormat(tensor->GetStorageFormat());
    const bool isFp4 = dtype == op::DataType::DT_FLOAT4_E2M1 || dtype == op::DataType::DT_FLOAT4_E1M2;
    const bool isNz = format == op::Format::FORMAT_FRACTAL_NZ || format == op::Format::FORMAT_FRACTAL_NZ_C0_32;
    if (isFp4 && isNz && storageShape.GetDimNum() > 0 && storageShape.GetDim(storageShape.GetDimNum() - 1) == 32) {
        if (storageElements > std::numeric_limits<int64_t>::max() / 2) {
            return ReportInvalidTensorStorage(tensor, opName, name, "packed FP4 storage element count overflows int64");
        }
        storageElements *= 2;
    }
    if (offset >= storageElements) {
        return ReportInvalidTensorStorage(tensor, opName, name, "offset is outside storage");
    }
    // maxIndex = offset + sum((viewShape[i] - 1) * strides[i]).
    // Subtract from the remaining capacity so neither multiplication nor accumulation can overflow.
    int64_t remaining = storageElements - 1 - offset;
    for (size_t i = 0; i < viewShape.GetDimNum(); ++i) {
        const int64_t steps = viewShape.GetDim(i) - 1;
        if (steps != 0 && strides[i] > remaining / steps) {
            return ReportInvalidTensorStorage(tensor, opName, name, "view shape, strides and offset exceed storage");
        }
        remaining -= steps * strides[i];
    }
    return true;
}

inline bool CheckTensorStorageBounds(const aclTensor* tensor, const char* opName, const char* name)
{
    if (tensor == nullptr) {
        return true;
    }
    return CheckTensorStorageBoundsWithViewShape(tensor, tensor->GetViewShape(), opName, name);
}

inline bool CheckTensorListStorageBounds(const aclTensorList* tensors, const char* opName, const char* name)
{
    if (tensors == nullptr) {
        return true;
    }
    for (size_t i = 0; i < tensors->Size(); ++i) {
        const std::string indexedName = std::string(name) + "[" + std::to_string(i) + "]";
        if (!CheckTensorStorageBounds((*tensors)[i], opName, indexedName.c_str())) {
            return false;
        }
    }
    return true;
}
} // namespace gmm

#endif // GMM_TENSOR_STORAGE_CHECK_H
