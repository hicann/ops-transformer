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
import sys
from typing import Optional

import torch
from torch.library import impl
from cann_ops_transformer.op_builder import OpBuilder, get_as_library

# experimental 算子不在 cann_ops_transformer 主包内，csrc 以本文件位置定位
# （builder 对 sources 的相对路径按主包 _package_path 解析，直接给绝对路径）
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# csrc 定位兼容两种布局: 源码树(本目录 csrc/) 与 whl 安装(包根 csrc/mhc/)
_PREMIX_CSRC = next(
    p
    for p in (
        os.path.join(_THIS_DIR, "csrc", "mhc_pre_sinkhorn_premix.cpp"),
        os.path.join(
            _THIS_DIR, "..", "..", "..", "csrc", "mhc", "mhc_pre_sinkhorn_premix.cpp"
        ),
    )
    if os.path.exists(p)
)


class MhcPreSinkhornPremixFunction(torch.autograd.Function):
    """Autograd wrapper: forward -> mhc_pre_sinkhorn_premix, backward -> mhc_pre_sinkhorn_premix_backward

    forward 返回 (hin, h_post, h_res, h_pre)：h_pre 暴露为输出以接入下游计算图
    （典型用法：作为下一轮调用的 premix）。backward 的 gradPre 即外部链路对
    h_pre 的梯度（grad_h_pre），由 autograd 图自动回传；h_pre 未被下游使用时
    由 autograd 物化为零张量（materialize_grads 默认行为）。
    """

    @staticmethod
    def forward(ctx, x, phi, alpha, bias, premix, hc_mult, num_iters, hc_eps, norm_eps):
        # autograd 路径强制 out_flag=True，以保存 backward 所需的中间结果
        op_module = mhc_pre_sinkhorn_premix_op_builder.load()
        hin, h_post, h_res, h_pre, hc_before_norm, inv_rms, sum_out, norm_out = (
            op_module.mhc_pre_sinkhorn_premix(
                x, phi, alpha, bias, premix, hc_mult, num_iters, hc_eps, norm_eps, True
            )
        )

        ctx.save_for_backward(
            x,
            phi,
            alpha,
            bias,
            premix,
            h_pre,
            hc_before_norm,
            inv_rms,
            sum_out,
            norm_out,
        )
        ctx.hc_eps = hc_eps
        ctx.hc_mult = hc_mult

        return hin, h_post, h_res, h_pre

    @staticmethod
    def backward(ctx, grad_hin, grad_h_post, grad_h_res, grad_h_pre):
        (
            x,
            phi,
            alpha,
            bias,
            premix,
            h_pre,
            hc_before_norm,
            inv_rms,
            sum_out,
            norm_out,
        ) = ctx.saved_tensors
        hc_eps = ctx.hc_eps
        n = ctx.hc_mult

        # 形状适配: forward 输出 h_res 为 flat (..., n*n)，autograd 回传同形 flat 梯度；
        # backward 算子契约 grad_h_res 为 (..., n, n)（与 verify/input.py 口径一致），
        # view 不改变内存布局，flat(n*n) 与 (n,n) row-major 位序相同。
        # 判据用维度数而非 shape[-2] != n：后者在 s == n（如 S=4=N）时漏判
        if (
            grad_h_res is not None
            and grad_h_res.dim() <= 3
            and grad_h_res.shape[-1] == n * n
        ):
            grad_h_res = grad_h_res.view(*grad_h_res.shape[:-1], n, n)

        # 反向复用 experimental 算子 mhc_pre_sinkhorn_premix_backward:
        # 优先 whl 嵌套包导入(site-packages 布局), 失败则回退平铺 torch_extension 导入
        try:
            from cann_ops_transformer_custom.ops.mhc.mhc_pre_sinkhorn_premix_backward.mhc_pre_sinkhorn_premix_backward import (  # noqa: E501
                mhc_pre_sinkhorn_premix_backward,
            )
        except ImportError:
            _bwd_torch_ext = os.path.abspath(
                os.path.join(
                    _THIS_DIR,
                    "..",
                    "..",
                    "..",
                    "mhc_pre_sinkhorn_premix_backward",
                    "torch_extension",
                )
            )
            if _bwd_torch_ext not in sys.path:
                sys.path.insert(0, _bwd_torch_ext)
            from mhc_pre_sinkhorn_premix_backward import (
                mhc_pre_sinkhorn_premix_backward,
            )

        # 契约（aclnn/tiling 层绑定校验）：premix 与 gradPre 必须同传或同缺省。
        # gradPre 为外部链路对本层 hPre 的梯度（grad_h_pre，经 autograd 图回传，
        # h_pre 未被下游使用时物化为零张量）；premix 未传入时二者同缺省。
        # 注：v1（premix 未传）轮的 h_pre 若被下游使用，其梯度因契约无入口而被
        # 丢弃（backward 算子不支持 preMix 缺省 + gradPre 传入的组合）。
        if premix is None:
            grad_pre = None
        else:
            grad_pre = (
                grad_h_pre if grad_h_pre is not None else torch.zeros_like(premix)
            )
        grad_x, grad_phi, grad_alpha, grad_bias, grad_premix = (
            mhc_pre_sinkhorn_premix_backward(
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
                premix,
                grad_pre,
                hc_eps,
            )
        )

        # 与 forward 参数一一对应，非 Tensor 参数的梯度填 None；
        # premix 的梯度为 grad_premix（未传 premix 时为空 tensor，归一化为 None）
        grad_premix_ret = (
            grad_premix if (premix is not None and grad_premix.numel() > 0) else None
        )
        return (
            grad_x,
            grad_phi,
            grad_alpha,
            grad_bias,
            grad_premix_ret,
            None,
            None,
            None,
            None,
        )


class MhcPreSinkhornPremixOpBuilder(OpBuilder):
    def __init__(self):
        super(MhcPreSinkhornPremixOpBuilder, self).__init__(
            "mhc_pre_sinkhorn_premix", category="mhc"
        )

    def sources(self):
        return [_PREMIX_CSRC]

    def schema(self) -> str:
        return (
            "mhc_pre_sinkhorn_premix(Tensor x, Tensor phi, Tensor alpha, Tensor bias, Tensor? premix, "
            "int hcMult, int numIters, float hcEps, float normEps, bool outFlag) -> "
            "(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def mhc_pre_sinkhorn_premix_meta(
            x, phi, alpha, bias, premix, hc_mult, num_iters, hc_eps, norm_eps, out_flag
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
                    # needGrad=false 但传入 premix 时也要输出 hPre
                    h_pre = (
                        torch.empty(t, n, dtype=phi.dtype, device="meta")
                        if premix is not None
                        else torch.empty(0, dtype=phi.dtype, device="meta")
                    )
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
                    # needGrad=false 但传入 premix 时也要输出 hPre
                    h_pre = (
                        torch.empty(b, s, n, dtype=phi.dtype, device="meta")
                        if premix is not None
                        else torch.empty(0, dtype=phi.dtype, device="meta")
                    )
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


mhc_pre_sinkhorn_premix_op_builder = MhcPreSinkhornPremixOpBuilder()
mhc_pre_sinkhorn_premix_op_builder._ensure_initialized()


@impl(get_as_library(), mhc_pre_sinkhorn_premix_op_builder.name, "PrivateUse1")
def _mhc_pre_sinkhorn_premix_dispatch(
    x, phi, alpha, bias, premix, hc_mult, num_iters, hc_eps, norm_eps, out_flag
):
    op_module = mhc_pre_sinkhorn_premix_op_builder.load()
    return op_module.mhc_pre_sinkhorn_premix(
        x, phi, alpha, bias, premix, hc_mult, num_iters, hc_eps, norm_eps, out_flag
    )


def mhc_pre_sinkhorn_premix(
    x,
    phi,
    alpha,
    bias,
    premix: Optional[torch.Tensor] = None,
    hc_mult: int = 4,
    num_iters: int = 20,
    hc_eps: float = 1e-6,
    norm_eps: float = 1e-6,
    need_backward: bool = False,
):
    """MhcPreSinkhornPremix 算子接口（experimental）。

    - premix: 可选输入，shape 为 x.shape[:-1]（即 (t, n) 或 (b, s, n)），dtype float32。
      传入时 hin 使用 premix 作为加权系数（通常填上一轮输出的 h_pre），而非本轮内部计算的 hPre。
      A2/A5 均支持。
    - need_backward=True 时，额外返回 h_pre/hc_before_norm/inv_rms/sum_out/norm_out 中间变量
      （无论是否传入 premix，均输出本轮内部计算的 h_pre，可供反向使用），返回 8 元组。
    - 若任一输入（x/phi/alpha/bias/premix）requires_grad=True 且未显式指定 need_backward，
      自动走 autograd 路径，返回 (hin, h_post, h_res, h_pre)：h_pre 为本轮内部计算值，
      可作为下一轮调用的 premix 接入梯度链（下一轮的 gradPreMix 会被 autograd 自动
      回传为本轮 backward 的 gradPre）。反向梯度复用 experimental 算子
      mhc_pre_sinkhorn_premix_backward。
    - 无梯度需求且 need_backward=False 时走普通路径（out_flag=False，premix 传入时 kernel
      仍会计算并写出 h_pre，但本接口不返回），返回 (hin, h_post, h_res)。
    """
    if need_backward:
        # 显式请求中间变量，返回 8 元组
        op_module = mhc_pre_sinkhorn_premix_op_builder.load()
        return op_module.mhc_pre_sinkhorn_premix(
            x, phi, alpha, bias, premix, hc_mult, num_iters, hc_eps, norm_eps, True
        )

    needs_grad = (
        x.requires_grad
        or phi.requires_grad
        or alpha.requires_grad
        or bias.requires_grad
        or (premix is not None and premix.requires_grad)
    )
    if needs_grad:
        # autograd 路径：out_flag=True，中间变量由 ctx.save_for_backward 保存，
        # 返回 4 元组（h_pre 可接入下游梯度链）
        return MhcPreSinkhornPremixFunction.apply(
            x, phi, alpha, bias, premix, hc_mult, num_iters, hc_eps, norm_eps
        )

    # 普通路径：out_flag=False，只返回 3 个主输出
    op_module = mhc_pre_sinkhorn_premix_op_builder.load()
    hin, h_post, h_res, _, _, _, _, _ = op_module.mhc_pre_sinkhorn_premix(
        x, phi, alpha, bias, premix, hc_mult, num_iters, hc_eps, norm_eps, False
    )
    return hin, h_post, h_res
