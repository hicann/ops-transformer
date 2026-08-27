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


class MhcPostBackwardOpBuilder(OpBuilder):
    def __init__(self):
        super(MhcPostBackwardOpBuilder, self).__init__(
            "mhc_post_backward", category="mhc"
        )

    def sources(self):
        return ["csrc/mhc/mhc_post_backward.cpp"]

    def schema(self) -> str:
        return (
            "mhc_post_backward(Tensor gradOutput, Tensor x, Tensor? hRes, Tensor hOut, Tensor hPost) -> "
            "(Tensor, Tensor, Tensor, Tensor)"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def mhc_post_backward_meta(grad_output, x, h_res, h_out, h_post):
            grad_x = torch.empty_like(x)
            if h_res is None or h_res.numel() == 0:
                grad_h_res = torch.empty(
                    *x.shape[:-1], x.shape[-2], dtype=torch.float32, device=x.device
                )
            else:
                grad_h_res = torch.empty_like(h_res)
            grad_h_out = torch.empty_like(h_out)
            grad_h_post = torch.empty_like(h_post)
            return (grad_x, grad_h_res, grad_h_out, grad_h_post)


mhc_post_backward_op_builder = MhcPostBackwardOpBuilder()
mhc_post_backward_op_builder._ensure_initialized()


@impl(get_as_library(), mhc_post_backward_op_builder.name, "PrivateUse1")
def _mhc_post_backward_dispatch(grad_output, x, h_res, h_out, h_post):
    op_module = mhc_post_backward_op_builder.load()
    return op_module.mhc_post_backward(grad_output, x, h_res, h_out, h_post)


def mhc_post_backward(grad_output, x, h_res, h_out, h_post):
    if h_res is None:
        h_res = torch.empty(0, dtype=torch.float32, device=grad_output.device)
    return torch.ops.cann_ops_transformer.mhc_post_backward(
        grad_output, x, h_res, h_out, h_post
    )
