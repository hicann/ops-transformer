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
    """V1 接口: 与合入前行为一致(无 pre_mix/grad_pre/enable_hf32), 固定 HF32 cube"""

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


class MhcPreSinkhornBackwardV2OpBuilder(OpBuilder):
    """V2 接口: 可选 pre_mix/grad_pre(必须成对)与 hc_eps(缺省 1e-6) + enable_hf32 属性下发"""

    def __init__(self):
        super(MhcPreSinkhornBackwardV2OpBuilder, self).__init__(
            "mhc_pre_sinkhorn_backward_v2", category="mhc"
        )

    def sources(self):
        return ["csrc/mhc/mhc_pre_sinkhorn_backward.cpp"]

    def schema(self) -> str:
        return (
            "mhc_pre_sinkhorn_backward_v2(Tensor gradHin, Tensor gradHPost, Tensor gradHRes, "
            "Tensor x, Tensor phi, Tensor alpha, Tensor bias, "
            "Tensor hPre, Tensor hcBeforeNorm, Tensor invRms, "
            "Tensor sumOut, Tensor normOut, *, "
            "Tensor? pre_mix=None, Tensor? grad_pre=None, float hc_eps=1e-6, bool enable_hf32=True) -> "
            "(Tensor, Tensor, Tensor, Tensor, Tensor)"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def mhc_pre_sinkhorn_backward_v2_meta(
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
            pre_mix=None,
            grad_pre=None,
            hc_eps=1e-6,
            enable_hf32=True,
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


mhc_pre_sinkhorn_backward_v2_op_builder = MhcPreSinkhornBackwardV2OpBuilder()
mhc_pre_sinkhorn_backward_v2_op_builder._ensure_initialized()


def _resolve_enable_hf32(enable_hf32):
    """enable_hf32 缺省(None)时跟随全局开关 torch_npu.npu.aclnn.allow_hf32。

    torch_npu 不可用或读取失败时回退为 True。显式传入 True/False 时优先使用显式值。
    """
    if enable_hf32 is not None:
        return enable_hf32
    try:
        import torch_npu

        return bool(torch_npu.npu.aclnn.allow_hf32)
    except Exception:
        return True


# meta/dispatch/converter 的形参名必须与 schema 参数名(pre_mix/grad_pre/enable_hf32)保持一致:
# 图模式下 keyword-only 参数(含 schema 默认值物化)按名以 kwargs 传入这些注册函数,
# 改成 snake_case 会导致图模式报 unexpected keyword argument。
@impl(get_as_library(), mhc_pre_sinkhorn_backward_v2_op_builder.name, "PrivateUse1")
def _mhc_pre_sinkhorn_backward_v2_dispatch(
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
    pre_mix=None,
    grad_pre=None,
    hc_eps=1e-6,
    enable_hf32=True,
):
    enable_hf32 = _resolve_enable_hf32(enable_hf32)
    op_module = mhc_pre_sinkhorn_backward_v2_op_builder.load()
    return op_module.mhc_pre_sinkhorn_backward_v2(
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
        pre_mix,
        grad_pre,
        enable_hf32,
    )


def _mhc_pre_sinkhorn_backward_v2(
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
    *,
    pre_mix=None,
    grad_pre=None,
    hc_eps=1e-6,
    enable_hf32=None,
):
    """V2 反向内部接口（供 mhc_pre_sinkhorn autograd 调用，不直接暴露给用户）。

    - pre_mix/grad_pre 必须成对传入或同时缺省
    - enable_hf32 缺省时跟随全局开关 torch_npu.npu.aclnn.allow_hf32
    """
    return _mhc_pre_sinkhorn_backward_v2_dispatch(
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
        _resolve_enable_hf32(enable_hf32),
    )
