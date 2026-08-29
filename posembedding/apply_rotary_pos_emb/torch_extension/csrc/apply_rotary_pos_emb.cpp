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
 * \file apply_rotary_pos_emb.cpp
 * \brief PyTorch extension wrapper for aclnnApplyRotaryPosEmbV2.
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {
namespace {
constexpr int64_t LAYOUT_BSND_BSH = 1;
constexpr int64_t LAYOUT_SBND = 2;
constexpr int64_t LAYOUT_BNSD = 3;
constexpr int64_t LAYOUT_TND = 4;
} // namespace

std::tuple<at::Tensor, at::Tensor> ApplyRotaryPosEmb(const at::Tensor &query, const at::Tensor &key,
                                                     const at::Tensor &cos, const at::Tensor &sin,
                                                     c10::string_view layout, c10::string_view rotary_mode)
{
    std::string layout_str = std::string(layout);
    std::string rotary_mode_str = std::string(rotary_mode);
    TORCH_CHECK(rotary_mode_str == "half" || rotary_mode_str == "quarter" || rotary_mode_str == "interleave",
                "apply_rotary_pos_emb: rotary_mode should be half/quarter/interleave, but got ", rotary_mode_str, ".");

    int64_t layout_value = LAYOUT_BSND_BSH;
    if (layout_str == "BNSD") {
        layout_value = LAYOUT_BNSD;
    } else if (layout_str == "SBND") {
        layout_value = LAYOUT_SBND;
    } else if (layout_str == "TND") {
        layout_value = LAYOUT_TND;
    }

    at::Tensor query_out = query.clone();
    at::Tensor key_out = key.clone();

    char *rotary_mode_ptr = const_cast<char *>(rotary_mode_str.c_str());
    ACLNN_CMD(aclnnApplyRotaryPosEmbV2, query_out, key_out, cos, sin, layout_value, rotary_mode_ptr);

    return std::make_tuple(query_out, key_out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("apply_rotary_pos_emb", &ApplyRotaryPosEmb, "apply_rotary_pos_emb");
}
} // namespace op_api
