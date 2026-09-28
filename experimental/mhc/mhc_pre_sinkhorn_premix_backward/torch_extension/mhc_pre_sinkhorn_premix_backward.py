# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import os

import torch
from torch.library import impl
from cann_ops_transformer.op_builder import OpBuilder, get_as_library

# experimental 算子不在 cann_ops_transformer 主包内，csrc 以本文件位置定位
# （builder 对 sources 的相对路径按主包 _package_path 解析，直接给绝对路径）
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# csrc 定位兼容两种布局: 源码树(本目录 csrc/) 与 whl 安装(包根 csrc/mhc/)
_PREMIX_BACKWARD_CSRC = next(
    p
    for p in (
        os.path.join(_THIS_DIR, "csrc", "mhc_pre_sinkhorn_premix_backward.cpp"),
        os.path.join(
            _THIS_DIR,
            "..",
            "..",
            "..",
            "csrc",
            "mhc",
            "mhc_pre_sinkhorn_premix_backward.cpp",
        ),
    )
    if os.path.exists(p)
)


class MhcPreSinkhornPremixBackwardOpBuilder(OpBuilder):
    def __init__(self):
        super(MhcPreSinkhornPremixBackwardOpBuilder, self).__init__(
            "mhc_pre_sinkhorn_premix_backward", category="mhc"
        )

    def sources(self):
        return [_PREMIX_BACKWARD_CSRC]

    def schema(self) -> str:
        return (
            "mhc_pre_sinkhorn_premix_backward(Tensor gradHin, Tensor gradHPost, Tensor gradHRes, "
            "Tensor x, Tensor phi, Tensor alpha, Tensor bias, "
            "Tensor hPre, Tensor hcBeforeNorm, Tensor invRms, "
            "Tensor sumOut, Tensor normOut, Tensor? preMix, Tensor? gradPre, float hcEps) -> "
            "(Tensor, Tensor, Tensor, Tensor, Tensor)"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def mhc_pre_sinkhorn_premix_backward_meta(
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
            pre_mix,
            grad_pre,
            hc_eps,
        ):
            grad_x = torch.empty_like(x)
            grad_phi = torch.empty_like(phi)
            grad_alpha = torch.empty_like(alpha)
            grad_bias = torch.empty_like(bias)
            # grad_pre_mix 仅在传入 pre_mix 时输出，shape 同 pre_mix（fp32）
            grad_pre_mix = (
                torch.empty_like(pre_mix)
                if pre_mix is not None
                else torch.empty(0, dtype=phi.dtype, device="meta")
            )
            return (grad_x, grad_phi, grad_alpha, grad_bias, grad_pre_mix)


mhc_pre_sinkhorn_premix_backward_op_builder = MhcPreSinkhornPremixBackwardOpBuilder()
mhc_pre_sinkhorn_premix_backward_op_builder._ensure_initialized()


@impl(get_as_library(), mhc_pre_sinkhorn_premix_backward_op_builder.name, "PrivateUse1")
def mhc_pre_sinkhorn_premix_backward(
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
    pre_mix,
    grad_pre,
    hc_eps,
):
    op_module = mhc_pre_sinkhorn_premix_backward_op_builder.load()
    return op_module.mhc_pre_sinkhorn_premix_backward(
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
        pre_mix,
        grad_pre,
        hc_eps,
    )
