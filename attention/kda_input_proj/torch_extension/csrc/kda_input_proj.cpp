/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <torch/extension.h>
#include "aclnn_common.h"

namespace op_api {
namespace {

constexpr int64_t DIM_TWO = 2;

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> ConstructKdaInputProjOutputs(const at::Tensor &x,
                                                                                        const at::Tensor &weight_qkv,
                                                                                        const at::Tensor &weight_beta,
                                                                                        const at::Tensor &weight_gate,
                                                                                        const at::Tensor &weight_g)
{
    TORCH_CHECK(x.dim() == DIM_TWO, "x must be 2D [T, hidden], but got ", x.dim(), "D.");
    TORCH_CHECK(x.size(0) > 0 && x.size(1) > 0, "All values within x's shape should be greater than 0.");
    TORCH_CHECK(weight_qkv.dim() == DIM_TWO && weight_beta.dim() == DIM_TWO && weight_gate.dim() == DIM_TWO &&
                    weight_g.dim() == DIM_TWO,
                "weights must be 2D matmul RHS [K, N].");
    const int64_t t_size = x.size(0);
    at::Tensor qkv;
    at::Tensor beta;
    at::Tensor gate;
    at::Tensor g;
    {
        const c10::OptionalDeviceGuard device_guard(c10::Device(x.device()));
        qkv = at::empty({t_size, weight_qkv.size(1)}, x.options().dtype(at::kBFloat16));
        beta = at::empty({t_size, weight_beta.size(1)}, x.options().dtype(at::kFloat));
        gate = at::empty({t_size, weight_gate.size(1)}, x.options().dtype(at::kBFloat16));
        g = at::empty({t_size, weight_g.size(1)}, x.options().dtype(at::kBFloat16));
    }
    return {qkv, beta, gate, g};
}

} // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> kda_input_proj(
    const at::Tensor &x, const at::Tensor &weight_qkv, const at::Tensor &weight_beta, const at::Tensor &weight_gate,
    const at::Tensor &weight_g, const at::Tensor &weight_qkv_scale)
{
    TORCH_CHECK(x.numel() > 0, "Tensor x is empty.");
    TORCH_CHECK(x.scalar_type() == at::ScalarType::BFloat16, "x dtype must be bfloat16, but got ", x.scalar_type(),
                ".");
    TORCH_CHECK(weight_beta.scalar_type() == at::ScalarType::BFloat16, "weight_beta dtype must be bfloat16.");
    TORCH_CHECK(weight_gate.scalar_type() == at::ScalarType::BFloat16, "weight_gate dtype must be bfloat16.");
    TORCH_CHECK(weight_g.scalar_type() == at::ScalarType::BFloat16, "weight_g dtype must be bfloat16.");
    TORCH_CHECK(weight_qkv_scale.defined(), "weight_qkv_scale is required and must be a defined tensor.");

    auto outputs = ConstructKdaInputProjOutputs(x, weight_qkv, weight_beta, weight_gate, weight_g);
    at::Tensor qkv = std::get<0>(outputs);
    at::Tensor beta = std::get<1>(outputs);
    at::Tensor gate = std::get<2>(outputs);
    at::Tensor g = std::get<3>(outputs);

    ACLNN_CMD(aclnnKdaInputProj, x, weight_qkv, weight_beta, weight_gate, weight_g, weight_qkv_scale, qkv, beta, gate,
              g);
    return {qkv, beta, gate, g};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("kda_input_proj", &kda_input_proj, "kda_input_proj");
}

} // namespace op_api
