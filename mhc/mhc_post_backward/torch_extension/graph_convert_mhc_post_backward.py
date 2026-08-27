# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# GE Converter for Graph Mode

try:
    import torch
    import torch_npu
    import torchair
    from typing import Optional
    from torchair.ge._ge_graph import Tensor, TensorSpec, DataType
    from torchair.ge._ge_graph import auto_convert_to_tensor
    from torchair._ge_concrete_graph.compat_ir import ge_op, IrDef
    from torchair._ge_concrete_graph.fx2ge_converter import (
        register_fx_node_ge_converter,
    )
    from torchair.ge import attr
    from torchair.ge._ge_graph import Const

    _TORCHAIR_AVAILABLE = True
except ImportError:
    _TORCHAIR_AVAILABLE = False

if _TORCHAIR_AVAILABLE:

    @auto_convert_to_tensor(
        [False, False, False, False, False], [False, False, True, False, False]
    )
    def mhc_post_backward_ge(
        grad_output: Tensor,
        x: Tensor,
        h_res: Optional[Tensor],
        h_out: Tensor,
        h_post: Tensor,
        *,
        dependencies=None,
        node_name=None,
    ):
        """REG_OP(MhcPostBackward)\n
        .INPUT(grad_y, TensorType({DT_BF16, DT_FLOAT16}))\n
        .INPUT(x, TensorType({DT_BF16, DT_FLOAT16}))\n
        .INPUT(h_res, TensorType({DT_FLOAT}))\n
        .INPUT(h_out, TensorType({DT_BF16, DT_FLOAT16}))\n
        .INPUT(h_post, TensorType({DT_FLOAT}))\n
        .OUTPUT(grad_x, TensorType({DT_BF16, DT_FLOAT16}))\n
        .OUTPUT(grad_h_res, TensorType({DT_FLOAT}))\n
        .OUTPUT(grad_h_out, TensorType({DT_BF16, DT_FLOAT16}))\n
        .OUTPUT(grad_h_post, TensorType({DT_FLOAT}))\n
        """
        if dependencies is None:
            dependencies = []
        if h_res is None:
            h_res = Const([], dtype=DataType.DT_FLOAT)
        inputs = {
            "grad_y": grad_output,
            "x": x,
            "h_res": h_res,
            "h_out": h_out,
            "h_post": h_post,
        }

        attrs = {}

        outputs = [
            "grad_x",
            "grad_h_res",
            "grad_h_out",
            "grad_h_post",
        ]

        return ge_op(
            op_type="MhcPostBackward",
            inputs=inputs,
            attrs=attrs,
            outputs=outputs,
            dependencies=dependencies,
            ir=IrDef("MhcPostBackward")
            .input("grad_y", "DT_BF16, DT_FLOAT16")
            .input("x", "DT_BF16, DT_FLOAT16")
            .input("h_res", "DT_FLOAT")
            .input("h_out", "DT_BF16, DT_FLOAT16")
            .input("h_post", "DT_FLOAT")
            .output("grad_x", "DT_BF16, DT_FLOAT16")
            .output("grad_h_res", "DT_FLOAT")
            .output("grad_h_out", "DT_BF16, DT_FLOAT16")
            .output("grad_h_post", "DT_FLOAT"),
        )

    @register_fx_node_ge_converter(
        torch.ops.cann_ops_transformer.mhc_post_backward.default
    )
    def convert_mhc_post_backward(
        grad_output: Tensor,
        x: Tensor,
        h_res: Optional[Tensor],
        h_out: Tensor,
        h_post: Tensor,
        meta_outputs: TensorSpec = None,
    ):
        return mhc_post_backward_ge(
            grad_output=grad_output, x=x, h_res=h_res, h_out=h_out, h_post=h_post
        )
