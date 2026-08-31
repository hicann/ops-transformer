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


class MhcPostFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, h_res, h_out, h_post):
        out = torch.ops.cann_ops_transformer.mhc_post(x, h_res, h_out, h_post)
        ctx.save_for_backward(x, h_res, h_out, h_post)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, h_res, h_out, h_post = ctx.saved_tensors
        from ..mhc_post_backward import mhc_post_backward

        # 0-stride expanded grad_output (e.g. from sum().backward()) breaks
        # aclnnMhcPostBackward when h_res is None; materialize it first.
        grad_output = grad_output.contiguous()
        grad_x, grad_h_res, grad_h_out, grad_h_post = mhc_post_backward(
            grad_output, x, h_res, h_out, h_post
        )
        if h_res is None:
            grad_h_res = None
        return grad_x, grad_h_res, grad_h_out, grad_h_post


class MhcPostOpBuilder(OpBuilder):
    def __init__(self):
        super(MhcPostOpBuilder, self).__init__("mhc_post", category="mhc")

    def sources(self):
        return ["csrc/mhc/mhc_post.cpp"]

    def schema(self) -> str:
        return "mhc_post(Tensor x, Tensor? hRes, Tensor hOut, Tensor hPost) -> Tensor"

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def mhc_post_meta(x, h_res, h_out, h_post):
            return torch.empty_like(x)


mhc_post_op_builder = MhcPostOpBuilder()
mhc_post_op_builder._ensure_initialized()


@impl(get_as_library(), mhc_post_op_builder.name, "PrivateUse1")
def _mhc_post_dispatch(x, h_res, h_out, h_post):
    op_module = mhc_post_op_builder.load()
    return op_module.mhc_post(x, h_res, h_out, h_post)


def mhc_post(x, h_res, h_out, h_post):
    needs_grad = (
        x.requires_grad
        or (h_res is not None and h_res.requires_grad)
        or h_out.requires_grad
        or h_post.requires_grad
    )

    if needs_grad:
        return MhcPostFunction.apply(x, h_res, h_out, h_post)
    else:
        return torch.ops.cann_ops_transformer.mhc_post(x, h_res, h_out, h_post)
