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
 * \file mhc_pre_sinkhorn_backward.cpp
 * \brief ACLNN Wrapper for aclnnMhcPreSinkhornBackward / aclnnMhcPreSinkhornBackwardV2
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {

// V1 接口: 与合入前行为一致(无 premix/gradPre/enableHf32), 固定 HF32 cube
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> MhcPreSinkhornBackward(
    const at::Tensor& gradHin, const at::Tensor& gradHPost, const at::Tensor& gradHRes, const at::Tensor& x,
    const at::Tensor& phi, const at::Tensor& alpha, const at::Tensor& bias, const at::Tensor& hPre,
    const at::Tensor& hcBeforeNorm, const at::Tensor& invRms, const at::Tensor& sumOut, const at::Tensor& normOut,
    double hcEps)
{
    at::Tensor gradX = at::empty_like(x);
    at::Tensor gradPhi = at::empty_like(phi);
    at::Tensor gradAlpha = at::empty_like(alpha);
    at::Tensor gradBias = at::empty_like(bias);

    ACLNN_CMD(aclnnMhcPreSinkhornBackward, gradHin, gradHPost, gradHRes, x, phi, alpha, bias, hPre, hcBeforeNorm,
              invRms, sumOut, normOut, hcEps, gradX, gradPhi, gradAlpha, gradBias);

    return std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>(gradX, gradPhi, gradAlpha, gradBias);
}

// V2 接口: 可选 premix/gradPre(必须成对) + enableHf32 属性下发
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor> MhcPreSinkhornBackwardV2(
    const at::Tensor& gradHin, const at::Tensor& gradHPost, const at::Tensor& gradHRes, const at::Tensor& x,
    const at::Tensor& phi, const at::Tensor& alpha, const at::Tensor& bias, const at::Tensor& hPre,
    const at::Tensor& hcBeforeNorm, const at::Tensor& invRms, const at::Tensor& sumOut, const at::Tensor& normOut,
    double hcEps, const c10::optional<at::Tensor>& preMix, const c10::optional<at::Tensor>& gradPre, bool enableHf32)
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

    ACLNN_CMD(aclnnMhcPreSinkhornBackwardV2, gradHin, gradHPost, gradHRes, x, phi, alpha, bias, hPre, hcBeforeNorm,
              invRms, sumOut, normOut, preMix, gradPre, hcEps, enableHf32, gradX, gradPhi, gradAlpha, gradBias,
              gradPreMix);

    return std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>(
        gradX, gradPhi, gradAlpha, gradBias,
        gradPreMix.has_value() ? gradPreMix.value() : at::empty({0}, phi.options()));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("mhc_pre_sinkhorn_backward", &MhcPreSinkhornBackward, "mhc_pre_sinkhorn_backward");
    // V2 末三参带默认值: 13 实参调用形态与 V1 完全等价(无 premix、HF32 开启),
    // V1 直调迁移到 V2 只需改函数名; preMix/gradPre 默认成对为 None, 语义安全。
    m.def("mhc_pre_sinkhorn_backward_v2", &MhcPreSinkhornBackwardV2, "mhc_pre_sinkhorn_backward_v2", py::arg("gradHin"),
          py::arg("gradHPost"), py::arg("gradHRes"), py::arg("x"), py::arg("phi"), py::arg("alpha"), py::arg("bias"),
          py::arg("hPre"), py::arg("hcBeforeNorm"), py::arg("invRms"), py::arg("sumOut"), py::arg("normOut"),
          py::arg("hcEps"), py::arg("preMix") = py::none(), py::arg("gradPre") = py::none(),
          py::arg("enableHf32") = true);
}

} // namespace op_api
