# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# GE Converter for Graph Mode
UNQUANT_MODE = -1
MXFP4_QUANT_MODE = 9
HIF4_QUANT_MODE = 10
INT4_DYNAMIC_QUANT_MODE = 13

# 4bit 输出在 GEIR 侧按逻辑元素描述维度与 dtype，PyTorch 侧用 uint8 承载打包数据
# （末维折半），需在 GE 图内补 pack 节点并显式声明 dtype 才能对齐 NetOutput。
# 取值为 (expanded_x dtype, expanded_scale dtype)，口径同 torchair PR #3134。
_PACKED_OUTPUT_DTYPES = {
    MXFP4_QUANT_MODE: ("DT_FLOAT4_E2M1", "DT_FLOAT8_E8M0"),
    HIF4_QUANT_MODE: ("DT_HIFLOAT4", "DT_FLOAT"),
    INT4_DYNAMIC_QUANT_MODE: ("DT_INT4", "DT_FLOAT"),
}

try:
    import torch
    import torch_npu
    import torchair
    from torchair._ge_concrete_graph import ge_apis as ge
    from torchair.ge._ge_graph import Tensor, TensorSpec
    from torchair._ge_concrete_graph.fx2ge_converter import (
        register_fx_node_ge_converter,
    )
    from torchair._ge_concrete_graph.supported_declaration import Support
    from torchair._ge_concrete_graph.compat_ir import ge_op, IrDef
    from torchair._ge_concrete_graph.utils import dtype_promote
    from torchair.ge._ge_graph import DataType, auto_convert_to_tensor
    from torchair.ge._ge_graph import get_default_ge_graph, next_unique_name
    from torchair.ge._ge_graph import (
        _ge_dtype_to_ge_proto_dtype,
        torch_dtype_value_to_ge_proto_type,
        torch_dtype_value_to_ge_type,
    )
    from torchair.ge import attr
    from typing import Optional, Union

    _TORCHAIR_AVAILABLE = True
except ImportError:
    _TORCHAIR_AVAILABLE = False

if _TORCHAIR_AVAILABLE:

    def set_ge_dtype(tensor: Tensor, dtype_name: str) -> None:
        """GEIR 推不出的量化 dtype 在此显式声明；torchair 无该 dtype 时跳过。"""
        ge_dtype = getattr(DataType, dtype_name, None)
        if ge_dtype is None:
            return
        tensor.desc.dtype = _ge_dtype_to_ge_proto_dtype(ge_dtype)

    def pack_4bit_to_uint8(tensor: Tensor) -> Tensor:
        """把 GE 图中的 4bit 逻辑张量折半打包成 uint8，实现同 torchair PR #3134。"""
        div_x2 = ge.Cast(ge.Const([1, 2]), dst_type=DataType.DT_INT32)
        packed_shape = ge.Div(ge.Shape(tensor), div_x2)
        logical_shape = ge.ConcatV2(
            [packed_shape, ge.Cast(ge.Const([2]), dst_type=DataType.DT_INT32)],
            concat_dim=0,
            N=2,
        )
        packed = ge.Bitcast(ge.Reshape(tensor, logical_shape), type=DataType.DT_UINT8)
        return ge.Reshape(packed, packed_shape)

    def is_mxfp4_passthrough_x(x: Tensor, x_dtype, quant_mode: int) -> bool:
        """非量化直通，且 x 以 uint8 承载 mxfp4（DT_FLOAT4_E2M1）打包数据。

        这是本算子 def 中唯一注册的 4bit 输入 dtype（HiFloat4/INT4 仅作输出），
        输入需解包、输出需 pack，两侧共用此判据；其余 dtype 一律不处理。
        """
        return (
            quant_mode == UNQUANT_MODE
            and x.dtype == DataType.DT_UINT8
            and x_dtype is not None
            and torch_dtype_value_to_ge_type(x_dtype) == DataType.DT_FLOAT4_E2M1
        )

    def unpack_mxfp4_x(x: Tensor, x_dtype) -> Tensor:
        """把 uint8 打包的 mxfp4 还原成 4bit 逻辑张量（末维 ×2）。"""
        unpacked = ge.Bitcast(x, type=torch_dtype_value_to_ge_type(x_dtype))
        unpacked.desc.dtype = torch_dtype_value_to_ge_proto_type(x_dtype)
        logical_shape = ge.Mul(ge.Shape(x), ge.Const([1] * (x.rank - 1) + [2]))
        return ge.Reshape(unpacked, logical_shape)

    def is_hifloat8_passthrough_x(x: Tensor, x_dtype, quant_mode: int) -> bool:
        """Identify HiFloat8 input carried by uint8 in the Torch interface."""
        return (
            quant_mode == UNQUANT_MODE
            and x.dtype == DataType.DT_UINT8
            and x_dtype is not None
            and torch_dtype_value_to_ge_type(x_dtype) == DataType.DT_HIFLOAT8
        )

    @auto_convert_to_tensor(
        [False, False, False, False, False, False],
        [False, False, True, True, True, True],
    )
    def MoeInitRoutingV4(
        x: Tensor,
        expert_idx: Tensor,
        scale: Optional[Tensor],
        offset: Optional[Tensor],
        active_num: Optional[Tensor],
        topk_weight: Optional[Tensor],
        *,
        expert_capacity: int,
        expert_num: int,
        drop_pad_mode: int,
        expert_tokens_num_type: int,
        expert_tokens_num_flag: bool,
        quant_mode: int,
        active_expert_range: Optional[list],
        row_idx_type: int,
        scale_alg: int = 0,
        dependencies=None,
        node_name=None,
    ):
        """REG_OP(MoeInitRoutingV4)\n
        .INPUT(x, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT, DT_INT8}))\n
        .INPUT(expert_idx, TensorType({DT_INT32}))\n
        .OPTIONAL_INPUT(scale, TensorType({DT_FLOAT}))\n
        .OPTIONAL_INPUT(offset, TensorType({DT_FLOAT}))\n
        .OPTIONAL_INPUT(active_num, TensorType({DT_INT64}))\n
        .OPTIONAL_INPUT(topk_weight, TensorType({DT_FLOAT}))\n
        .OUTPUT(expanded_x, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT, DT_INT8, DT_HIFLOAT4}))\n
        .OUTPUT(expanded_row_idx, TensorType({DT_INT32}))\n
        .OUTPUT(expert_tokens_count_or_cumsum, TensorType({DT_INT64}))\n
        .OUTPUT(expanded_scale, TensorType({DT_FLOAT}))\n
        .OUTPUT(expanded_topk_weight, TensorType({DT_FLOAT}))\n
        .ATTR(expert_capacity, Int, -1)\n
        .ATTR(expert_num, Int, -1)\n
        .ATTR(drop_pad_mode, Int, 0)\n
        .ATTR(expert_tokens_num_type, Int, 0)\n
        .ATTR(expert_tokens_num_flag, Bool, False)\n
        .ATTR(quant_mode, Int, -1)\n
        .ATTR(active_expert_range, ListInt, {-1, -1})\n
        .ATTR(row_idx_type, Int, 0)\n
        .ATTR(scale_alg, Int, 0)\n
        """
        if dependencies is None:
            dependencies = []

        inputs = {
            "x": x,
            "expert_idx": expert_idx,
            "scale": scale,
            "offset": offset,
            "active_num": active_num,
            "topk_weight": topk_weight,
        }

        attrs = {
            "expert_capacity": attr.Int(expert_capacity),
            "expert_num": attr.Int(expert_num),
            "drop_pad_mode": attr.Int(drop_pad_mode),
            "expert_tokens_num_type": attr.Int(expert_tokens_num_type),
            "expert_tokens_num_flag": attr.Bool(expert_tokens_num_flag),
            "quant_mode": attr.Int(quant_mode),
            "row_idx_type": attr.Int(row_idx_type),
            "scale_alg": attr.Int(scale_alg),
        }
        if active_expert_range is not None:
            attrs["active_expert_range"] = attr.ListInt(active_expert_range)
        else:
            attrs["active_expert_range"] = attr.ListInt([])

        outputs = [
            "expanded_x",
            "expanded_row_idx",
            "expert_tokens_count_or_cumsum",
            "expanded_scale",
            "expanded_topk_weight",
        ]

        return ge_op(
            op_type="MoeInitRoutingV4",
            inputs=inputs,
            attrs=attrs,
            outputs=outputs,
            dependencies=dependencies,
            ir=IrDef("MoeInitRoutingV4")
            .input(
                "x",
                "DT_INT8, DT_FLOAT16, DT_BF16, DT_FLOAT, DT_HIFLOAT8, "
                "DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT4_E2M1",
            )
            .input("expert_idx", "DT_INT32")
            .optional_input("scale", "DT_FLOAT, DT_FLOAT8_E8M0")
            .optional_input("offset", "DT_FLOAT")
            .optional_input("active_num", "DT_INT64")
            .optional_input("topk_weight", "DT_FLOAT")
            .attr("expert_capacity", attr.Int(-1))
            .attr("expert_num", attr.Int(-1))
            .attr("drop_pad_mode", attr.Int(0))
            .attr("expert_tokens_num_type", attr.Int(0))
            .attr("expert_tokens_num_flag", attr.Bool(False))
            .attr("quant_mode", attr.Int(-1))
            .attr("active_expert_range", attr.ListInt([]))
            .attr("row_idx_type", attr.Int(0))
            .attr("scale_alg", attr.Int(0))
            .output(
                "expanded_x",
                "DT_INT8, DT_FLOAT16, DT_BF16, DT_FLOAT, DT_FLOAT8_E5M2, "
                "DT_FLOAT8_E4M3FN, DT_HIFLOAT8, DT_FLOAT4_E2M1, DT_INT4, DT_HIFLOAT4",
            )
            .output("expanded_row_idx", "DT_INT32")
            .output("expert_tokens_count_or_cumsum", "DT_INT64")
            .output("expanded_scale", "DT_FLOAT, DT_FLOAT8_E8M0")
            .output("expanded_topk_weight", "DT_FLOAT"),
        )

    @register_fx_node_ge_converter(
        torch.ops.cann_ops_transformer.moe_init_routing.default
    )
    def convert_moe_init_routing(
        x: Tensor,
        expert_idx: Tensor,
        *,
        scale: Tensor = None,
        offset: Tensor = None,
        topk_weight: Tensor = None,
        active_num: Union[int, Tensor] = -1,
        expert_capacity: int = -1,
        expert_num: int = -1,
        drop_pad_mode: int = 0,
        expert_tokens_num_type: int = 0,
        expert_tokens_num_flag: bool = False,
        quant_mode: int = -1,
        active_expert_range: list = None,
        row_idx_type: int = 0,
        x_dtype: int = None,
        scale_alg: int = 0,
        meta_outputs: TensorSpec = None,
    ):
        active_num_tensor = dtype_promote(active_num, target_dtype=DataType.DT_INT64)
        mxfp4_passthrough = is_mxfp4_passthrough_x(x, x_dtype, quant_mode)
        hifloat8_passthrough = is_hifloat8_passthrough_x(x, x_dtype, quant_mode)
        if mxfp4_passthrough:
            x = unpack_mxfp4_x(x, x_dtype)
        elif hifloat8_passthrough:
            x = ge.Bitcast(x, type=DataType.DT_HIFLOAT8)
            x.desc.dtype = torch_dtype_value_to_ge_proto_type(x_dtype)

        (
            expanded_x,
            expanded_row_idx,
            expert_tokens_count_or_cumsum,
            expanded_scale,
            expanded_topk_weight,
        ) = MoeInitRoutingV4(
            x=x,
            expert_idx=expert_idx,
            scale=scale,
            offset=offset,
            active_num=active_num_tensor,
            topk_weight=topk_weight,
            expert_capacity=expert_capacity,
            expert_num=expert_num,
            drop_pad_mode=drop_pad_mode,
            expert_tokens_num_type=expert_tokens_num_type,
            expert_tokens_num_flag=expert_tokens_num_flag,
            quant_mode=quant_mode,
            active_expert_range=active_expert_range,
            row_idx_type=row_idx_type,
            scale_alg=scale_alg,
        )

        if quant_mode in _PACKED_OUTPUT_DTYPES:
            expanded_x_dtype, expanded_scale_dtype = _PACKED_OUTPUT_DTYPES[quant_mode]
            set_ge_dtype(expanded_x, expanded_x_dtype)
            set_ge_dtype(expanded_scale, expanded_scale_dtype)
            expanded_x = pack_4bit_to_uint8(expanded_x)
        elif mxfp4_passthrough:
            expanded_x = pack_4bit_to_uint8(expanded_x)
        elif hifloat8_passthrough:
            expanded_x = ge.Bitcast(expanded_x, type=DataType.DT_UINT8)

        return (
            expanded_x,
            expanded_row_idx,
            expert_tokens_count_or_cumsum,
            expanded_scale,
            expanded_topk_weight,
        )
