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
 * \file mhc_pre_sinkhorn_premix_backward.cpp
 * \brief ACLNN Wrapper for aclnnMhcPreSinkhornPremixBackward
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> MhcPreSinkhornPremixBackward(
    const at::Tensor &gradHin, const at::Tensor &gradHPost, const at::Tensor &gradHRes, const at::Tensor &x,
    const at::Tensor &phi, const at::Tensor &alpha, const at::Tensor &bias, const at::Tensor &hPre,
    const at::Tensor &hcBeforeNorm, const at::Tensor &invRms, const at::Tensor &sumOut, const at::Tensor &normOut,
    const c10::optional<at::Tensor> &preMix, const c10::optional<at::Tensor> &gradPre, double hcEps)
{
    at::Tensor gradX = at::empty_like(x);
    at::Tensor gradPhi = at::empty_like(phi);
    at::Tensor gradAlpha = at::empty_like(alpha);
    at::Tensor gradBias = at::empty_like(bias);
    // grad_pre_mix 仅在传入 pre_mix 时计算，shape 同 pre_mix（fp32 [B,S,N] / [T,N]）
    c10::optional<at::Tensor> gradPreMix;
    if (preMix.has_value()) {
        gradPreMix = at::empty_like(preMix.value());
    }

    ACLNN_CMD(aclnnMhcPreSinkhornPremixBackward, gradHin, gradHPost, gradHRes, x, phi, alpha, bias, hPre, hcBeforeNorm,
              invRms, sumOut, normOut, preMix, gradPre, hcEps, gradX, gradPhi, gradAlpha, gradBias, gradPreMix);

    return std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>(
        gradX, gradPhi, gradAlpha, gradBias,
        gradPreMix.has_value() ? gradPreMix.value() : at::empty({0}, phi.options()));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("mhc_pre_sinkhorn_premix_backward", &MhcPreSinkhornPremixBackward, "mhc_pre_sinkhorn_premix_backward");
}

} // namespace op_api
