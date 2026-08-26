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


class MhcPreSinkhornFunction(torch.autograd.Function):
    """Autograd wrapper: forward -> mhc_pre_sinkhorn, backward -> mhc_pre_sinkhorn_backward"""

    @staticmethod
    def forward(ctx, x, phi, alpha, bias, hc_mult, num_iters, hc_eps, norm_eps):
        # autograd 路径强制 out_flag=True，以保存 backward 所需的中间结果
        op_module = mhc_pre_sinkhorn_op_builder.load()
        hin, h_post, h_res, h_pre, hc_before_norm, inv_rms, sum_out, norm_out = (
            op_module.mhc_pre_sinkhorn(
                x, phi, alpha, bias, hc_mult, num_iters, hc_eps, norm_eps, True
            )
        )

        ctx.save_for_backward(
            x, phi, alpha, bias, h_pre, hc_before_norm, inv_rms, sum_out, norm_out
        )
        ctx.hc_eps = hc_eps

        return hin, h_post, h_res

    @staticmethod
    def backward(ctx, grad_hin, grad_h_post, grad_h_res):
        x, phi, alpha, bias, h_pre, hc_before_norm, inv_rms, sum_out, norm_out = (
            ctx.saved_tensors
        )
        hc_eps = ctx.hc_eps

        from ..mhc_pre_sinkhorn_backward import (
            mhc_pre_sinkhorn_backward,
        )

        grad_x, grad_phi, grad_alpha, grad_bias = mhc_pre_sinkhorn_backward(
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

        # 与 forward 参数一一对应，非 Tensor 参数的梯度填 None
        return grad_x, grad_phi, grad_alpha, grad_bias, None, None, None, None


class MhcPreSinkhornOpBuilder(OpBuilder):
    def __init__(self):
        super(MhcPreSinkhornOpBuilder, self).__init__(
            "mhc_pre_sinkhorn", category="mhc"
        )

    def sources(self):
        return ["csrc/mhc/mhc_pre_sinkhorn.cpp"]

    def schema(self) -> str:
        return (
            "mhc_pre_sinkhorn(Tensor x, Tensor phi, Tensor alpha, Tensor bias, "
            "int hcMult, int numIters, float hcEps, float normEps, bool outFlag) -> "
            "(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def mhc_pre_sinkhorn_meta(
            x, phi, alpha, bias, hc_mult, num_iters, hc_eps, norm_eps, out_flag
        ):
            n = hc_mult
            if x.dim() == 3:
                t = x.size(0)
                c = x.size(2)
                hin = torch.empty(t, c, dtype=x.dtype, device="meta")
                h_post = torch.empty(t, n, dtype=phi.dtype, device="meta")
                h_res = torch.empty(t, n * n, dtype=phi.dtype, device="meta")
                if out_flag:
                    h_pre = torch.empty(t, n, dtype=phi.dtype, device="meta")
                    hc_before_norm = torch.empty(
                        t, n * n + 2 * n, dtype=phi.dtype, device="meta"
                    )
                    inv_rms = torch.empty(t, 1, dtype=phi.dtype, device="meta")
                    sum_out = torch.empty(
                        2 * num_iters, t, n, dtype=phi.dtype, device="meta"
                    )
                    norm_out = torch.empty(
                        2 * num_iters, t, n, n, dtype=phi.dtype, device="meta"
                    )
                else:
                    h_pre = torch.empty(0, dtype=phi.dtype, device="meta")
                    hc_before_norm = torch.empty(0, dtype=phi.dtype, device="meta")
                    inv_rms = torch.empty(0, dtype=phi.dtype, device="meta")
                    sum_out = torch.empty(0, dtype=phi.dtype, device="meta")
                    norm_out = torch.empty(0, dtype=phi.dtype, device="meta")
            else:
                b = x.size(0)
                s = x.size(1)
                c = x.size(3)
                hin = torch.empty(b, s, c, dtype=x.dtype, device="meta")
                h_post = torch.empty(b, s, n, dtype=phi.dtype, device="meta")
                h_res = torch.empty(b, s, n * n, dtype=phi.dtype, device="meta")
                if out_flag:
                    h_pre = torch.empty(b, s, n, dtype=phi.dtype, device="meta")
                    hc_before_norm = torch.empty(
                        b, s, n * n + 2 * n, dtype=phi.dtype, device="meta"
                    )
                    inv_rms = torch.empty(b, s, 1, dtype=phi.dtype, device="meta")
                    sum_out = torch.empty(
                        2 * num_iters, b, s, n, dtype=phi.dtype, device="meta"
                    )
                    norm_out = torch.empty(
                        2 * num_iters, b, s, n, n, dtype=phi.dtype, device="meta"
                    )
                else:
                    h_pre = torch.empty(0, dtype=phi.dtype, device="meta")
                    hc_before_norm = torch.empty(0, dtype=phi.dtype, device="meta")
                    inv_rms = torch.empty(0, dtype=phi.dtype, device="meta")
                    sum_out = torch.empty(0, dtype=phi.dtype, device="meta")
                    norm_out = torch.empty(0, dtype=phi.dtype, device="meta")

            return (
                hin,
                h_post,
                h_res,
                h_pre,
                hc_before_norm,
                inv_rms,
                sum_out,
                norm_out,
            )


mhc_pre_sinkhorn_op_builder = MhcPreSinkhornOpBuilder()
mhc_pre_sinkhorn_op_builder._ensure_initialized()


@impl(get_as_library(), mhc_pre_sinkhorn_op_builder.name, "PrivateUse1")
def _mhc_pre_sinkhorn_dispatch(
    x, phi, alpha, bias, hc_mult, num_iters, hc_eps, norm_eps, out_flag
):
    op_module = mhc_pre_sinkhorn_op_builder.load()
    return op_module.mhc_pre_sinkhorn(
        x, phi, alpha, bias, hc_mult, num_iters, hc_eps, norm_eps, out_flag
    )


def mhc_pre_sinkhorn(
    x, phi, alpha, bias, hc_mult, num_iters, hc_eps=1e-6, norm_eps=1e-6
):
    """自动反向传播接口，正向始终只返回 (hin, h_post, h_res)。

    - 若任一输入设置了 requires_grad=True，走 autograd 路径，自动保存中间变量用于 backward。
    - 若无输入需梯度，走普通路径，中间变量不保存（节省内存）。
    - backward 通过 tensor.grad 自动获取梯度。
    """
    needs_grad = (
        x.requires_grad
        or phi.requires_grad
        or alpha.requires_grad
        or bias.requires_grad
    )

    if needs_grad:
        # autograd 路径：out_flag=True，中间变量由 ctx.save_for_backward 保存
        return MhcPreSinkhornFunction.apply(
            x, phi, alpha, bias, hc_mult, num_iters, hc_eps, norm_eps
        )
    else:
        # 普通路径：out_flag=False，不保存中间变量，只返回 3 个主输出
        op_module = mhc_pre_sinkhorn_op_builder.load()
        hin, h_post, h_res, _, _, _, _, _ = op_module.mhc_pre_sinkhorn(
            x, phi, alpha, bias, hc_mult, num_iters, hc_eps, norm_eps, False
        )
        return hin, h_post, h_res
