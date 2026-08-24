/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file inplace_partial_rotary_mul_backward.cpp
 * \brief ACLNN Wrapper for aclnnInplacePartialRotaryMulGrad
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {

static const std::unordered_map<std::string, int64_t> mode_map = {
    {"half", 0}, {"interleave", 1}, {"quarter", 2}, {"interleave-half", 3}};

void InplacePartialRotaryMulBackward(at::Tensor &gradOutput, const at::Tensor &r1, const at::Tensor &r2,
                                     std::string rotaryMode, c10::IntArrayRef partialSlice)
{
    TORCH_CHECK(gradOutput.dim() == 4, "Input tensor grad_output's dim num should be 4, actual ", gradOutput.dim(),
                ".");
    auto it = mode_map.find(rotaryMode);
    TORCH_CHECK(it != mode_map.end(),
                "rotary_mode must be one of 'half', 'interleave', 'quarter', 'interleave-half', got '", rotaryMode,
                "'.");
    TORCH_CHECK(rotaryMode == "interleave", "rotary_mode only supports 'interleave', got '", rotaryMode, "'.");
    int64_t modeInt = it->second;

    TORCH_CHECK(partialSlice.size() == 2, "partial_slice must contain exactly two integers.");
    int64_t sliceStart = partialSlice[0];
    int64_t sliceEnd = partialSlice[1];
    TORCH_CHECK(sliceStart >= 0, "partial_slice start must be >= 0, got ", sliceStart, ".");
    TORCH_CHECK(sliceEnd >= 0, "partial_slice end must be >= 0, got ", sliceEnd, ".");
    TORCH_CHECK(sliceStart <= sliceEnd, "partial_slice start must be <= end, got start=", sliceStart,
                ", end=", sliceEnd, ".");
    TORCH_CHECK(sliceEnd <= gradOutput.size(3), "partial_slice end must be <= D (", gradOutput.size(3),
                "), got end=", sliceEnd, ".");
    TORCH_CHECK(gradOutput.is_contiguous(), "grad_output must be a contiguous tensor.");
    TORCH_CHECK(r1.is_contiguous(), "r1 must be a contiguous tensor.");
    TORCH_CHECK(r2.is_contiguous(), "r2 must be a contiguous tensor.");

    ACLNN_CMD(aclnnInplacePartialRotaryMulGrad, gradOutput, r1, r2, modeInt, partialSlice);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("inplace_partial_rotary_mul_backward", &InplacePartialRotaryMulBackward,
          "inplace_partial_rotary_mul_backward");
}
} // namespace op_api
