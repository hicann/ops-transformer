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
    from torch.library import impl
    from torchair._ge_concrete_graph import ge_apis as ge
    from torchair.ge._ge_graph import Tensor, TensorSpec
    from torchair._ge_concrete_graph.fx2ge_converter import (
        declare_supported,
        register_fx_node_ge_converter,
    )
    from torchair._ge_concrete_graph.supported_declaration import Support
    from typing import Any, Dict, List, Tuple, Union, Callable, Optional
    from torchair._ge_concrete_graph.ge_ir_pb2 import (
        GraphDef,
        OpDef,
        TensorDescriptor,
        TensorDef,
    )
    from torchair.ge._ge_graph import get_default_ge_graph, next_unique_name
    from torchair.ge._ge_graph import auto_convert_to_tensor
    from torchair.ge._ge_graph import Tensor, TensorSpec, DataType, TensorType
    from torchair.ge._ge_graph import compat_as_bytes, compat_as_bytes_list
    from torchair.ge._ge_graph import trans_to_list_list_int, trans_to_list_list_float
    from torchair.ge._ge_graph import get_invalid_desc
    from torchair._ge_concrete_graph.compat_ir import ge_op, IrDef
    from torchair.ge import attr

    _TORCHAIR_AVAILABLE = True
except ImportError:
    _TORCHAIR_AVAILABLE = False

if _TORCHAIR_AVAILABLE:

    @auto_convert_to_tensor([False, False, False, False], [False, False, False, False])
    def MhcPreSinkhorn(
        x: Tensor,
        phi: Tensor,
        alpha: Tensor,
        bias: Tensor,
        *,
        hc_mult: int = 4,
        num_iters: int = 20,
        hc_eps: float = 1e-6,
        norm_eps: float = 1e-6,
        need_backward: bool = True,
        dependencies=None,
        node_name=None,
    ):
        """REG_OP(MhcPreSinkhorn)\n
        .INPUT(x, TensorType({DT_BF16, DT_FLOAT16}))\n
        .INPUT(phi, TensorType({DT_FLOAT}))\n
        .INPUT(alpha, TensorType({DT_FLOAT}))\n
        .INPUT(bias, TensorType({DT_FLOAT}))\n
        .OUTPUT(hin, TensorType({DT_BF16, DT_FLOAT16}))\n
        .OUTPUT(hPost, TensorType({DT_FLOAT}))\n
        .OUTPUT(hRes, TensorType({DT_FLOAT}))\n
        .OUTPUT(hPre, TensorType({DT_FLOAT}))\n
        .OUTPUT(hcBeforeNorm, TensorType({DT_FLOAT}))\n
        .OUTPUT(invRms, TensorType({DT_FLOAT}))\n
        .OUTPUT(sumOut, TensorType({DT_FLOAT}))\n
        .OUTPUT(normOut, TensorType({DT_FLOAT}))\n
        .ATTR(hc_mult, Int, 4)\n
        .ATTR(num_iters, Int, 20)\n
        .ATTR(hc_eps, Float, 1e-6)\n
        .ATTR(norm_eps, Float, 1e-6)\n
        .ATTR(need_backward, Bool, true)\n
        """
        if dependencies is None:
            dependencies = []

        inputs = {
            "x": x,
            "phi": phi,
            "alpha": alpha,
            "bias": bias,
        }

        attrs = {
            "hc_mult": attr.Int(hc_mult),
            "num_iters": attr.Int(num_iters),
            "hc_eps": attr.Float(hc_eps),
            "norm_eps": attr.Float(norm_eps),
            "need_backward": attr.Bool(need_backward),
        }

        outputs = [
            "hin",
            "hPost",
            "hRes",
            "hPre",
            "hcBeforeNorm",
            "invRms",
            "sumOut",
            "normOut",
        ]

        return ge_op(
            op_type="MhcPreSinkhorn",
            inputs=inputs,
            attrs=attrs,
            outputs=outputs,
            dependencies=dependencies,
            ir=IrDef("MhcPreSinkhorn")
            .input("x", "DT_BF16, DT_FLOAT16")
            .input("phi", "DT_FLOAT")
            .input("alpha", "DT_FLOAT")
            .input("bias", "DT_FLOAT")
            .attr("hc_mult", attr.Int(4))
            .attr("num_iters", attr.Int(20))
            .attr("hc_eps", attr.Float(1e-6))
            .attr("norm_eps", attr.Float(1e-6))
            .attr("need_backward", attr.Bool(True))
            .output("hin", "DT_BF16, DT_FLOAT16")
            .output("hPost", "DT_FLOAT")
            .output("hRes", "DT_FLOAT")
            .output("hPre", "DT_FLOAT")
            .output("hcBeforeNorm", "DT_FLOAT")
            .output("invRms", "DT_FLOAT")
            .output("sumOut", "DT_FLOAT")
            .output("normOut", "DT_FLOAT"),
        )

    @register_fx_node_ge_converter(
        torch.ops.cann_ops_transformer.mhc_pre_sinkhorn.default
    )
    def convert_mhc_pre_sinkhorn(
        x: Tensor,
        phi: Tensor,
        alpha: Tensor,
        bias: Tensor,
        hc_mult: int,
        num_iters: int,
        hc_eps: float,
        norm_eps: float,
        out_flag: bool,
        meta_outputs: TensorSpec = None,
    ):
        return MhcPreSinkhorn(
            x=x,
            phi=phi,
            alpha=alpha,
            bias=bias,
            hc_mult=hc_mult,
            num_iters=num_iters,
            hc_eps=hc_eps,
            norm_eps=norm_eps,
            need_backward=out_flag,
        )
