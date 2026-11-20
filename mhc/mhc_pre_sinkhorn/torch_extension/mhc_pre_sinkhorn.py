# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import torch
from torch.library import impl
from cann_ops_transformer.op_builder import OpBuilder, get_as_library


class MhcPreSinkhornFunction(torch.autograd.Function):
    """Autograd wrapper: forward -> mhc_pre_sinkhorn, backward -> mhc_pre_sinkhorn_backward"""

    @staticmethod
    def forward(
        ctx,
        x,
        phi,
        alpha,
        bias,
        pre_mix,
        hc_mult,
        num_iters,
        hc_eps,
        norm_eps,
        enable_hf32,
    ):
        # autograd 路径强制 out_flag=True，以保存 backward 所需的中间结果
        op_module = mhc_pre_sinkhorn_op_builder.load()
        hin, h_post, h_res, h_pre, hc_before_norm, inv_rms, sum_out, norm_out = (
            op_module.mhc_pre_sinkhorn(
                x,
                phi,
                alpha,
                bias,
                hc_mult,
                num_iters,
                hc_eps,
                norm_eps,
                True,
                pre_mix,
                enable_hf32,
            )
        )

        ctx.save_for_backward(
            x,
            phi,
            alpha,
            bias,
            pre_mix,
            h_pre,
            hc_before_norm,
            inv_rms,
            sum_out,
            norm_out,
        )
        ctx.hc_eps = hc_eps
        ctx.enable_hf32 = enable_hf32

        if pre_mix is not None:
            # premix 分支透出 h_pre（sinkhorn 权重）：下游可消费，其梯度经
            # grad_h_pre -> grad_pre 回传给反向算子，契约与 experimental premix 接口一致
            return hin, h_post, h_res, h_pre
        return hin, h_post, h_res

    @staticmethod
    def backward(ctx, grad_hin, grad_h_post, grad_h_res, grad_h_pre=None):
        (
            x,
            phi,
            alpha,
            bias,
            pre_mix,
            h_pre,
            hc_before_norm,
            inv_rms,
            sum_out,
            norm_out,
        ) = ctx.saved_tensors
        hc_eps = ctx.hc_eps
        enable_hf32 = ctx.enable_hf32

        from ..mhc_pre_sinkhorn_backward.mhc_pre_sinkhorn_backward import (
            _mhc_pre_sinkhorn_backward_v2,
        )

        # grad_pre 为外部链路对本层 hPre 输出的梯度：forward 透出 h_pre 时由 autograd
        # 回传（未消费时为 None/零张量）。hin 链式项(Σ_c gradHin·x)由 backward kernel
        # 内部计入 gradPreMix 输出（kernel 输出名），wrapper 不得重复传入，否则会污染
        # SigmoidGrad(h_hat_pre) 链路的 grad_phi/grad_alpha/grad_bias。
        if pre_mix is not None:
            grad_pre = (
                grad_h_pre if grad_h_pre is not None else torch.zeros_like(pre_mix)
            )
        else:
            grad_pre = None
        grad_x, grad_phi, grad_alpha, grad_bias, grad_pre_mix = (
            _mhc_pre_sinkhorn_backward_v2(
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
                pre_mix=pre_mix,
                grad_pre=grad_pre,
                hc_eps=hc_eps,
                enable_hf32=enable_hf32,
            )
        )

        # 与 forward 参数一一对应，非 Tensor 参数的梯度填 None；
        # pre_mix 的梯度为 grad_pre_mix（未传 pre_mix 时为空 tensor，归一化为 None）
        grad_pre_mix_ret = (
            grad_pre_mix if (pre_mix is not None and grad_pre_mix.numel() > 0) else None
        )
        return (
            grad_x,
            grad_phi,
            grad_alpha,
            grad_bias,
            grad_pre_mix_ret,
            None,
            None,
            None,
            None,
            None,
        )


class MhcPreSinkhornOpBuilder(OpBuilder):
    def __init__(self):
        super(MhcPreSinkhornOpBuilder, self).__init__(
            "mhc_pre_sinkhorn", category="mhc"
        )

    def sources(self):
        return ["csrc/mhc/mhc_pre_sinkhorn.cpp"]

    def schema(self) -> str:
        # 参数面与公开 API 对齐, 不含 enable_hf32: HF32 由 dispatch 内部读取全局开关
        # torch_npu.npu.aclnn.allow_hf32 后下发给 aclnn(与公开 wrapper 同源同函数,
        # aclnn 层 enableHf32 属性接口不变)。pre_mix 必须为位置参数(带默认值):
        # 放在 * 之后会被 torch.library.register_autograd 的 has_kwarg_only_tensors
        # 检查拒绝; 全位置参数使 setup_context/backward 走最简三参契约
        # (setup_context(ctx=, inputs=, output=), 不传 keyword_only_inputs)。
        # 注意 torchair GE 侧构建函数(graph_convert 的 MhcPreSinkhorn)的 premix
        # 仍须 keyword-only, 其约束在 GE 函数签名上, 与本 schema 无关。
        return (
            "mhc_pre_sinkhorn(Tensor x, Tensor phi, Tensor alpha, Tensor bias, "
            "int hcMult, int numIters, float hcEps, float normEps, bool outFlag, "
            "Tensor? pre_mix=None) -> "
            "(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def mhc_pre_sinkhorn_meta(
            x,
            phi,
            alpha,
            bias,
            hc_mult,
            num_iters,
            hc_eps,
            norm_eps,
            out_flag,
            pre_mix=None,
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
                    # needGrad=false 但传入 pre_mix 时也要输出 hPre
                    h_pre = (
                        torch.empty(t, n, dtype=phi.dtype, device="meta")
                        if pre_mix is not None
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
                    # needGrad=false 但传入 pre_mix 时也要输出 hPre
                    h_pre = (
                        torch.empty(b, s, n, dtype=phi.dtype, device="meta")
                        if pre_mix is not None
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


mhc_pre_sinkhorn_op_builder = MhcPreSinkhornOpBuilder()
mhc_pre_sinkhorn_op_builder._ensure_initialized()


def _get_global_allow_hf32():
    """读取全局HF32开关 torch_npu.npu.aclnn.allow_hf32。

    torch_npu 不可用或读取失败时回退为 True（与合入前 HF32 行为一致）。
    公开 wrapper 与 PrivateUse1 dispatch 均由此取值下发 aclnn,
    保证两条路径的 HF32 语义一致。
    """
    try:
        import torch_npu

        return bool(torch_npu.npu.aclnn.allow_hf32)
    except Exception:
        return True


# meta/dispatch/converter 的形参名必须与 schema 参数名(pre_mix)保持一致:
# 图模式下 torchair 可按名以 kwargs 传入这些注册函数, 形参名不一致会报
# unexpected keyword argument。
@impl(get_as_library(), mhc_pre_sinkhorn_op_builder.name, "PrivateUse1")
def _mhc_pre_sinkhorn_dispatch(
    x,
    phi,
    alpha,
    bias,
    hc_mult,
    num_iters,
    hc_eps,
    norm_eps,
    out_flag,
    pre_mix=None,
):
    # schema 不含 enable_hf32: 与公开 wrapper 同源, 读全局开关后下发给 aclnn
    enable_hf32 = _get_global_allow_hf32()
    op_module = mhc_pre_sinkhorn_op_builder.load()
    return op_module.mhc_pre_sinkhorn(
        x,
        phi,
        alpha,
        bias,
        hc_mult,
        num_iters,
        hc_eps,
        norm_eps,
        out_flag,
        pre_mix,
        enable_hf32,
    )


def mhc_pre_sinkhorn(
    x, phi, alpha, bias, hc_mult, num_iters, hc_eps=1e-6, norm_eps=1e-6, *, pre_mix=None
):
    """自动反向传播接口。

    - 不传 pre_mix：与旧版一致，正向返回 (hin, h_post, h_res)。
    - 传入 pre_mix 且任一输入 requires_grad=True：返回 (hin, h_post, h_res, h_pre)，
      第 4 个输出 h_pre 为 sinkhorn 权重，可接入下游梯度链——其梯度经 grad_pre
      传给反向算子，契约与 experimental premix 接口一致；普通路径（无梯度需求）
      不透出 h_pre，仍返回 3 个主输出。
    - backward 通过 tensor.grad 自动获取梯度。
    - HF32 模式不作为参数暴露：每次调用自动读取全局开关 torch_npu.npu.aclnn.allow_hf32
      并随调用下发；autograd 反向沿用前向下发时的取值。
    """
    enable_hf32 = _get_global_allow_hf32()
    needs_grad = (
        x.requires_grad
        or phi.requires_grad
        or alpha.requires_grad
        or bias.requires_grad
        or (pre_mix is not None and pre_mix.requires_grad)
    )

    if needs_grad:
        # autograd 路径：out_flag=True，中间变量由 ctx.save_for_backward 保存
        return MhcPreSinkhornFunction.apply(
            x,
            phi,
            alpha,
            bias,
            pre_mix,
            hc_mult,
            num_iters,
            hc_eps,
            norm_eps,
            enable_hf32,
        )
    else:
        # 普通路径：out_flag=False，不保存中间变量，只返回 3 个主输出
        op_module = mhc_pre_sinkhorn_op_builder.load()
        hin, h_post, h_res, _, _, _, _, _ = op_module.mhc_pre_sinkhorn(
            x,
            phi,
            alpha,
            bias,
            hc_mult,
            num_iters,
            hc_eps,
            norm_eps,
            False,
            pre_mix,
            enable_hf32,
        )
        return hin, h_post, h_res
