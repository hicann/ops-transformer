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
    def MhcPost(
        x: Tensor,
        h_res: Tensor,
        h_out: Tensor,
        h_post: Tensor,
        *,
        dependencies=[],
        node_name=None,
    ):
        """REG_OP(MhcPost)\n
        .INPUT(x, TensorType({DT_BF16, DT_FLOAT16}))\n
        .INPUT(h_res, TensorType({DT_FLOAT}))\n
        .INPUT(h_out, TensorType({DT_BF16, DT_FLOAT16}))\n
        .INPUT(h_post, TensorType({DT_FLOAT}))\n
        .OUTPUT(y, TensorType({DT_BF16, DT_FLOAT16}))\n
        """
        inputs = {
            "x": x,
            "h_res": h_res,
            "h_out": h_out,
            "h_post": h_post,
        }

        attrs = {}

        outputs = [
            "y",
        ]

        return ge_op(
            op_type="MhcPost",
            inputs=inputs,
            attrs=attrs,
            outputs=outputs,
            dependencies=dependencies,
            ir=IrDef("MhcPost")
            .input("x", "DT_BF16, DT_FLOAT16")
            .input("h_res", "DT_FLOAT")
            .input("h_out", "DT_BF16, DT_FLOAT16")
            .input("h_post", "DT_FLOAT")
            .output("y", "DT_BF16, DT_FLOAT16"),
        )

    @register_fx_node_ge_converter(torch.ops.cann_ops_transformer.mhc_post.default)
    def convert_mhc_post(
        x: Tensor,
        hRes: Tensor,
        hOut: Tensor,
        hPost: Tensor,
        meta_outputs: TensorSpec = None,
    ):
        return MhcPost(x=x, h_res=hRes, h_out=hOut, h_post=hPost)
