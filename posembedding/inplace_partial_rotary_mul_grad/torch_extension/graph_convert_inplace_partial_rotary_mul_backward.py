# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ---------------------------------------------------------------------------
# GE Converter for Graph Mode
try:
    import torch
    import torch_npu
    import torchair
    from typing import List, Optional
    from torchair.ge._ge_graph import Tensor, TensorSpec
    from torchair.ge._ge_graph import auto_convert_to_tensor
    from torchair._ge_concrete_graph.compat_ir import ge_op, IrDef
    from torchair._ge_concrete_graph.fx2ge_converter import (
        register_fx_node_ge_converter,
    )
    from torchair.ge import attr

    _TORCHAIR_AVAILABLE = True
except ImportError:
    _TORCHAIR_AVAILABLE = False


_ROTARY_MODE_MAP = {
    "interleave": 1,
}


if _TORCHAIR_AVAILABLE:

    @auto_convert_to_tensor([False, False, False], [False, False, False])
    def inplace_partial_rotary_mul_backward(
        grad_output: Tensor,
        cos: Tensor,
        sin: Tensor,
        *,
        rotary_mode: int = 0,
        partial_slice: Optional[List[int]] = None,
        dependencies=None,
        node_name=None,
    ):
        """REG_OP(InplacePartialRotaryMulGrad)\n
        .INPUT(dy, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))\n
        .INPUT(cos, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))\n
        .INPUT(sin, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))\n
        .OUTPUT(dy, TensorType({DT_FLOAT16, DT_FLOAT, DT_BFLOAT16}))\n
        .ATTR(rotary_mode, Int, 0)\n
        .ATTR(partial_slice, ListInt, {0, 0})\n
        """
        if partial_slice is None:
            partial_slice = [0, 0]
        if dependencies is None:
            dependencies = []

        inputs = {
            "dy": grad_output,
            "cos": cos,
            "sin": sin,
        }
        attrs = {
            "rotary_mode": attr.Int(rotary_mode),
            "partial_slice": attr.ListInt(partial_slice),
        }
        outputs = ["dy"]

        return ge_op(
            op_type="InplacePartialRotaryMulGrad",
            inputs=inputs,
            attrs=attrs,
            outputs=outputs,
            dependencies=dependencies,
            node_name=node_name,
            ir=IrDef("InplacePartialRotaryMulGrad")
            .input("dy", "DT_FLOAT16, DT_FLOAT, DT_BFLOAT16")
            .input("cos", "DT_FLOAT16, DT_FLOAT, DT_BFLOAT16")
            .input("sin", "DT_FLOAT16, DT_FLOAT, DT_BFLOAT16")
            .attr("rotary_mode", attr.Int(0))
            .attr("partial_slice", attr.ListInt([0, 0]))
            .output("dy", "DT_FLOAT16, DT_FLOAT, DT_BFLOAT16"),
        )

    @register_fx_node_ge_converter(
        torch.ops.cann_ops_transformer.inplace_partial_rotary_mul_backward.default
    )
    def convert_inplace_partial_rotary_mul_backward(
        grad_output: Tensor,
        r1: Tensor,
        r2: Tensor,
        *,
        rotary_mode: str = "interleave",
        partial_slice: Optional[List[int]] = None,
        meta_outputs: TensorSpec = None,
    ):
        partial_slice = [0, 0] if partial_slice is None else partial_slice
        if rotary_mode not in _ROTARY_MODE_MAP:
            raise ValueError(
                f"rotary_mode only supports 'interleave', got '{rotary_mode}'."
            )
        return inplace_partial_rotary_mul_backward(
            grad_output=grad_output,
            cos=r1,
            sin=r2,
            rotary_mode=_ROTARY_MODE_MAP[rotary_mode],
            partial_slice=partial_slice,
        )
else:

    def convert_inplace_partial_rotary_mul_backward(*args, **kwargs):
        raise RuntimeError(
            "GE converter requires torchair, but torchair is not available."
        )
