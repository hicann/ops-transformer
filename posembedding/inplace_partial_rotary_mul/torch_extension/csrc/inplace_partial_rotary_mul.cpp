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
 * \file inplace_partial_rotary_mul.cpp
 * \brief
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {

static const std::unordered_map<std::string, int64_t> mode_map = {
    {"half", 0}, {"interleave", 1}, {"quarter", 2}, {"interleave-half", 3}};

void InplacePartialRotaryMul(at::Tensor &x, const at::Tensor &r1, const at::Tensor &r2, std::string rotaryMode,
                             c10::IntArrayRef partialSlice)
{
    TORCH_CHECK(x.dim() == 4, "Input tensor x's dim num should be 4, actual ", x.dim(), ".");
    auto it = mode_map.find(rotaryMode);
    TORCH_CHECK(it != mode_map.end(),
                "rotary_mode must be one of 'half', 'interleave', 'quarter', 'interleave-half', got '", rotaryMode,
                "'.");
    TORCH_CHECK(rotaryMode == "interleave", "rotary_mode only supports 'interleave', got '", rotaryMode, "'.");
    int64_t modeInt = it->second;

    ACLNN_CMD(aclnnInplacePartialRotaryMul, x, r1, r2, modeInt, partialSlice);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("inplace_partial_rotary_mul", &InplacePartialRotaryMul, "inplace_partial_rotary_mul");
}
} // namespace op_api
