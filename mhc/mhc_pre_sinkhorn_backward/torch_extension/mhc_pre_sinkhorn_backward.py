# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import torch
from torch.library import impl
from cann_ops_transformer.op_builder import OpBuilder, get_as_library


class MhcPreSinkhornBackwardOpBuilder(OpBuilder):
    def __init__(self):
        super(MhcPreSinkhornBackwardOpBuilder, self).__init__(
            "mhc_pre_sinkhorn_backward", category="mhc"
        )

    def sources(self):
        return ["csrc/mhc/mhc_pre_sinkhorn_backward.cpp"]

    def schema(self) -> str:
        return (
            "mhc_pre_sinkhorn_backward(Tensor gradHin, Tensor gradHPost, Tensor gradHRes, "
            "Tensor x, Tensor phi, Tensor alpha, Tensor bias, "
            "Tensor hPre, Tensor hcBeforeNorm, Tensor invRms, "
            "Tensor sumOut, Tensor normOut, float hcEps) -> "
            "(Tensor, Tensor, Tensor, Tensor)"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def mhc_pre_sinkhorn_backward_meta(
            grad_hin,
            grad_h_post,
            grad_h_res,
            x,
            phi,
            alpha,
            bias,
            h_pre,
            hc_before_norm,
            inv_rms,
            sum_out,
            norm_out,
            hc_eps,
        ):
            grad_x = torch.empty_like(x)
            grad_phi = torch.empty_like(phi)
            grad_alpha = torch.empty_like(alpha)
            grad_bias = torch.empty_like(bias)
            return (grad_x, grad_phi, grad_alpha, grad_bias)


mhc_pre_sinkhorn_backward_op_builder = MhcPreSinkhornBackwardOpBuilder()
mhc_pre_sinkhorn_backward_op_builder._ensure_initialized()


@impl(get_as_library(), mhc_pre_sinkhorn_backward_op_builder.name, "PrivateUse1")
def mhc_pre_sinkhorn_backward(
    grad_hin,
    grad_h_post,
    grad_h_res,
    x,
    phi,
    alpha,
    bias,
    h_pre,
    hc_before_norm,
    inv_rms,
    sum_out,
    norm_out,
    hc_eps,
):
    op_module = mhc_pre_sinkhorn_backward_op_builder.load()
    return op_module.mhc_pre_sinkhorn_backward(
        grad_hin,
        grad_h_post,
        grad_h_res,
        x,
        phi,
        alpha,
        bias,
        h_pre,
        hc_before_norm,
        inv_rms,
        sum_out,
        norm_out,
        hc_eps,
    )
