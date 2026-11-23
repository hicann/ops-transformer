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

import functools
import torch

try:
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


def _is_routing_input(node):
    # Keep the compatibility hook scoped to direct inputs of this operator.
    # Unsupported FP4 uses elsewhere retain TorchAir's original behavior.
    return (
        node is not None
        and bool(node.users)
        and all(
            str(user.target) == "cann_ops_transformer.moe_finalize_routing.default"
            and (
                (bool(user.args) and user.args[0] is node)
                or user.kwargs.get("expanded_x") is node
            )
            for user in node.users
        )
    )


def _aclgraph_routing_indices(fx_graph):
    if fx_graph is None:
        return ()
    return tuple(
        index
        for index, node in enumerate(
            node for node in fx_graph.graph.nodes if node.op == "placeholder"
        )
        if _is_routing_input(node)
    )


def _process_mxfp4_aclgraph_input(original, graph, graph_key, *args):
    indices = getattr(graph, "_moe_mxfp4_inputs", None)
    if indices is None:
        indices = _aclgraph_routing_indices(graph._fx_graph)
    indices = tuple(
        i
        for i in indices
        if isinstance(args[i], torch.Tensor) and args[i].dtype == torch.float4_e2m1fn_x2
    )
    if not indices:
        return original(graph, graph_key, *args)

    meta = graph._graphs_meta[graph_key]
    replay_args = list(args)
    restored = []
    try:
        for index in indices:
            source = args[index]
            if not source.is_contiguous():
                raise RuntimeError(
                    "MoeFinalizeRouting MXFP4 ACLGraph input must be contiguous"
                )
            replay_args[index] = source.view(torch.uint8)
            # Only process_input sees byte views. Capture/replay and alias metadata
            # retain native FP4 tensors, storage offsets and device addresses.
            for name in (
                "userinputs_metatensor",
                "retained_userinputs",
                "captured_userinput_ref_with_output",
            ):
                mapping = getattr(meta, name)
                if mapping and index in mapping:
                    tensor = mapping[index]
                    restored.append((mapping, index, tensor))
                    mapping[index] = tensor.view(torch.uint8)
        return original(graph, graph_key, *replay_args)
    finally:
        for mapping, index, tensor in reversed(restored):
            mapping[index] = tensor


def install_mxfp4_aclgraph_adapter():
    """Copy this operator's packed FP4 inputs as bytes before ACLGraph replay."""
    if getattr(torch, "float4_e2m1fn_x2", None) is None:
        return
    try:
        import torch_npu  # Registers the bundled npugraph_ex package.
        from npugraph_ex._acl_concrete_graph.acl_graph import AclGraph
        from npugraph_ex._acl_concrete_graph.fx2acl_converter import AclConcreteGraph
    except ImportError:
        return

    original = AclGraph.process_input
    if getattr(original, "_moe_mxfp4_adapter", False):
        return

    @functools.wraps(original)
    def process_input(graph, graph_key, *args):
        return _process_mxfp4_aclgraph_input(original, graph, graph_key, *args)

    # Cached ACLGraph execution builds a manager from generated forward code,
    # without an FX graph. Carry the operator-scoped input indices into that code.
    original_codegen = AclConcreteGraph._codegen_init

    @functools.wraps(original_codegen)
    def codegen_init(graph, extend_config):
        code = original_codegen(graph, extend_config)
        indices = _aclgraph_routing_indices(graph.fx_graph)
        return code + f"\nacl_graph._moe_mxfp4_inputs = {indices!r}\n"

    process_input._moe_mxfp4_adapter = True
    AclGraph.process_input = process_input
    AclConcreteGraph._codegen_init = codegen_init


install_mxfp4_aclgraph_adapter()


if _TORCHAIR_AVAILABLE:

    class _PackedByteInput:
        """TorchAir input processor, including its generated/cached execution path."""

        def __init__(self, index):
            self.fx_input_idx = index

        def __call__(self, *args):
            return args[self.fx_input_idx].view(torch.uint8)

        def codegen(self, ge_index, input_name):
            return (
                f"{input_name}[{ge_index}] = "
                f"args[{self.fx_input_idx}].view(torch.uint8)"
            )

    def _parse_packed_input(original, graph, target, args, kwargs, meta):
        fp4 = getattr(torch, "float4_e2m1fn_x2", None)
        node = getattr(getattr(graph, "_converter_ctx", None), "node", None)
        if (
            fp4 is None
            or not isinstance(meta, torch.Tensor)
            or meta.dtype != fp4
            or not _is_routing_input(node)
        ):
            return original(graph, target, args, kwargs, meta)
        if not meta.is_contiguous():
            raise RuntimeError(
                "MoeFinalizeRouting MXFP4 graph input must be contiguous"
            )

        index = graph.graph.num_inputs
        data = original(graph, target, args, kwargs, meta.view(torch.uint8))
        graph.graph.named_inputs_info[data.node.name].func = _PackedByteInput(index)
        # Preserve the public FX metadata and input guards, not the transport dtype.
        graph._all_meta_tensor_input[index] = meta
        data._moe_mxfp4_packed_input = True
        return data

    def install_mxfp4_input_adapter():
        """Install once, only when TorchAir lacks the native Torch FP4 mapping."""
        fp4 = getattr(torch, "float4_e2m1fn_x2", None)
        if fp4 is None:
            return
        from torchair.ge import _ge_graph
        from torchair._ge_concrete_graph.fx2ge_converter import GeConcreteGraph

        original = GeConcreteGraph.parse_input
        if getattr(original, "_moe_mxfp4_adapter", False):
            return
        try:
            _ge_graph.torch_type_to_ge_type(fp4)
        except RuntimeError:
            pass
        else:
            return  # Leave releases with native dtype support untouched.

        @functools.wraps(original)
        def parse_input(graph, target, args, kwargs, meta_outputs):
            return _parse_packed_input(
                original, graph, target, args, kwargs, meta_outputs
            )

        parse_input._moe_mxfp4_adapter = True
        GeConcreteGraph.parse_input = parse_input

    def restore_mxfp4_input(tensor):
        """Recover logical FP4 width from the packed byte input, also dynamically."""
        from torchair._ge_concrete_graph import ge_apis as ge
        from torchair.ge._ge_graph import DataType

        if tensor.dtype != DataType.DT_UINT8 and not getattr(
            tensor, "_moe_mxfp4_packed_input", False
        ):
            return tensor
        # Torch stores two FP4 numbers per element. GE describes logical elements.
        fp4_per_byte = 2
        multiples = ge.Const([1] * (tensor.rank - 1) + [fp4_per_byte])
        logical_shape = ge.Mul(ge.Shape(tensor), multiples)
        unpacked = ge.Bitcast(tensor, type=DataType.DT_FLOAT4_E2M1)
        return ge.Reshape(unpacked, logical_shape)

    install_mxfp4_input_adapter()

    @auto_convert_to_tensor(
        [
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        ],
        [False, False, True, True, True, True, True, True, True, True, True, True],
    )
    def MoeFinalizeRouting(
        expanded_x: Tensor,
        expanded_row_idx: Tensor,
        x1: Optional[Tensor],
        x2: Optional[Tensor],
        bias: Optional[Tensor],
        scales: Optional[Tensor],
        expert_idx: Optional[Tensor],
        x: Optional[Tensor],
        alpha1: Optional[Tensor],
        alpha2: Optional[Tensor],
        v: Optional[Tensor],
        expanded_x_scale: Optional[Tensor],
        *,
        drop_pad_mode: int = 0,
        zero_expert_range: Optional[List[int]] = None,
        copy_expert_range: Optional[List[int]] = None,
        constant_expert_range: Optional[List[int]] = None,
        k: int = 1,
        dependencies=None,
        node_name=None,
    ):
        """REG_OP(MoeFinalizeRoutingV2)\n
        .INPUT(expanded_x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16, DT_FLOAT4_E2M1}))\n
        .INPUT(expanded_row_idx, TensorType({DT_INT32}))\n
        .OPTIONAL_INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))\n
        .OPTIONAL_INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))\n
        .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))\n
        .OPTIONAL_INPUT(scales, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))\n
        .OPTIONAL_INPUT(expert_idx, TensorType({DT_INT32}))\n
        .OPTIONAL_INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))\n
        .OPTIONAL_INPUT(alpha1, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))\n
        .OPTIONAL_INPUT(alpha2, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))\n
        .OPTIONAL_INPUT(v, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))\n
        .OPTIONAL_INPUT(expanded_x_scale, TensorType({DT_FLOAT8_E8M0}))\n
        .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))\n
        .ATTR(drop_pad_mode, Int, 0)\n
        .ATTR(zero_expert_range, ListInt, {})\n
        .ATTR(copy_expert_range, ListInt, {})\n
        .ATTR(constant_expert_range, ListInt, {})\n
        .ATTR(k, Int, 1)\n
        """
        if zero_expert_range is None:
            zero_expert_range = []
        if copy_expert_range is None:
            copy_expert_range = []
        if constant_expert_range is None:
            constant_expert_range = []
        if dependencies is None:
            dependencies = []

        from torchair._ge_concrete_graph import ge_apis as ge
        from torchair.ge._ge_graph import DataType

        is_mxfp4 = expanded_x.dtype in (
            DataType.DT_UINT8,
            DataType.DT_FLOAT4_E2M1,
        ) or getattr(expanded_x, "_moe_mxfp4_packed_input", False)
        if is_mxfp4:
            from .moe_finalize_routing import _check_mxfp4_special_experts

            _check_mxfp4_special_experts(
                (x, alpha1, alpha2, v),
                (zero_expert_range, copy_expert_range, constant_expert_range),
            )
        expanded_x = restore_mxfp4_input(expanded_x)
        if (
            is_mxfp4
            and expanded_x_scale is not None
            and expanded_x_scale.dtype == DataType.DT_UINT8
        ):
            expanded_x_scale = ge.Bitcast(
                expanded_x_scale, type=DataType.DT_FLOAT8_E8M0
            )

        inputs = {
            "expanded_x": expanded_x,
            "expanded_row_idx": expanded_row_idx,
            "x1": x1,
            "x2": x2,
            "bias": bias,
            "scales": scales,
            "expert_idx": expert_idx,
            "x": x,
            "alpha1": alpha1,
            "alpha2": alpha2,
            "v": v,
            "expanded_x_scale": expanded_x_scale if is_mxfp4 else None,
        }

        attrs = {
            "drop_pad_mode": attr.Int(0 if drop_pad_mode is None else drop_pad_mode),
            "zero_expert_range": attr.ListInt(zero_expert_range),
            "copy_expert_range": attr.ListInt(copy_expert_range),
            "constant_expert_range": attr.ListInt(constant_expert_range),
            "k": attr.Int(1 if k is None else k),
        }

        outputs = ["y"]

        # The MXFP4 scale is input 11. Keep placeholders for omitted optional inputs
        # so GE cannot interpret the scale as an earlier tensor such as expert_idx.
        if is_mxfp4:
            return ge_op(
                op_type="MoeFinalizeRoutingV2",
                inputs=inputs,
                attrs=attrs,
                outputs=outputs,
                dependencies=dependencies,
                node_name=node_name,
            )

        return ge_op(
            op_type="MoeFinalizeRoutingV2",
            inputs=inputs,
            attrs=attrs,
            outputs=outputs,
            dependencies=dependencies,
            node_name=node_name,
            ir=IrDef("MoeFinalizeRoutingV2")
            .input("expanded_x", "DT_FLOAT, DT_FLOAT16, DT_BF16, DT_FLOAT4_E2M1")
            .input("expanded_row_idx", "DT_INT32")
            .optional_input("x1", "DT_FLOAT, DT_FLOAT16, DT_BF16")
            .optional_input("x2", "DT_FLOAT, DT_FLOAT16, DT_BF16")
            .optional_input("bias", "DT_FLOAT, DT_FLOAT16, DT_BF16")
            .optional_input("scales", "DT_FLOAT, DT_FLOAT16, DT_BF16")
            .optional_input("expert_idx", "DT_INT32")
            .optional_input("x", "DT_FLOAT, DT_FLOAT16, DT_BF16")
            .optional_input("alpha1", "DT_FLOAT, DT_FLOAT16, DT_BF16")
            .optional_input("alpha2", "DT_FLOAT, DT_FLOAT16, DT_BF16")
            .optional_input("v", "DT_FLOAT, DT_FLOAT16, DT_BF16")
            .optional_input("expanded_x_scale", "DT_FLOAT8_E8M0")
            .attr("drop_pad_mode", attr.Int(0))
            .attr("zero_expert_range", attr.ListInt([]))
            .attr("copy_expert_range", attr.ListInt([]))
            .attr("constant_expert_range", attr.ListInt([]))
            .attr("k", attr.Int(1))
            .output("y", "DT_FLOAT, DT_FLOAT16, DT_BF16"),
        )

    @register_fx_node_ge_converter(
        torch.ops.cann_ops_transformer.moe_finalize_routing.default
    )
    def convert_moe_finalize_routing(
        expanded_x: Tensor,
        expanded_row_idx: Tensor,
        x1: Optional[Tensor],
        x2: Optional[Tensor],
        bias: Optional[Tensor],
        scales: Optional[Tensor],
        expert_idx: Optional[Tensor],
        x: Optional[Tensor],
        alpha1: Optional[Tensor],
        alpha2: Optional[Tensor],
        v: Optional[Tensor],
        drop_pad_mode: int = 0,
        zero_expert_range: Optional[List[int]] = None,
        copy_expert_range: Optional[List[int]] = None,
        constant_expert_range: Optional[List[int]] = None,
        k: int = 1,
        *,
        expanded_x_scale: Optional[Tensor] = None,
        meta_outputs: TensorSpec = None,
    ):
        return MoeFinalizeRouting(
            expanded_x=expanded_x,
            expanded_row_idx=expanded_row_idx,
            x1=x1,
            x2=x2,
            bias=bias,
            scales=scales,
            expert_idx=expert_idx,
            x=x,
            alpha1=alpha1,
            alpha2=alpha2,
            v=v,
            expanded_x_scale=expanded_x_scale,
            drop_pad_mode=drop_pad_mode,
            zero_expert_range=zero_expert_range,
            copy_expert_range=copy_expert_range,
            constant_expert_range=constant_expert_range,
            k=k,
        )
