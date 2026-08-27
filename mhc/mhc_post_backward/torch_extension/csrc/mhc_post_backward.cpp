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
 * \file mhc_post_backward.cpp
 * \brief ACLNN Wrapper for aclnnMhcPostBackward
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {

constexpr int64_t DIMS_THREE = 3;
constexpr int64_t DIMS_FOUR = 4;
constexpr size_t SECOND_TO_LAST_OFFSET = 2;

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> MhcPostBackward(const at::Tensor &gradOutput,
                                                                           const at::Tensor &x,
                                                                           const c10::optional<at::Tensor> &hRes,
                                                                           const at::Tensor &hOut,
                                                                           const at::Tensor &hPost)
{
    const c10::OptionalDeviceGuard deviceGuard(x.device());
    bool hasHRes = hRes.has_value() && hRes.value().numel() > 0;
    at::Tensor gradX = at::empty_like(x);
    at::Tensor gradHres;
    if (hasHRes) {
        gradHres = at::empty_like(hRes.value());
    } else {
        TORCH_CHECK(x.dim() == DIMS_THREE || x.dim() == DIMS_FOUR,
                    "x must be 3D (TND) or 4D (BSND) when h_res is None or empty, but got ", x.dim(), "D");
        auto sizes = x.sizes().vec();
        sizes.back() = sizes[sizes.size() - SECOND_TO_LAST_OFFSET];
        gradHres = at::empty(sizes, x.options().dtype(at::kFloat));
    }
    at::Tensor gradHout = at::empty_like(hOut);
    at::Tensor gradHpost = at::empty_like(hPost);

    c10::optional<at::Tensor> hResArg = hasHRes ? c10::optional<at::Tensor>(hRes.value()) : c10::nullopt;
    ACLNN_CMD(aclnnMhcPostBackward, gradOutput, x, hResArg, hOut, hPost, gradX, gradHres, gradHout, gradHpost);

    return std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>(gradX, gradHres, gradHout, gradHpost);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("mhc_post_backward", &MhcPostBackward, "mhc_post_backward");
}

} // namespace op_api
