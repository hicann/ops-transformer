# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Optional TorchAir converter for AttentionWorkerCombine."""

try:
    import torch
    from torchair.ge import attr, DataType
    from torchair.ge._ge_graph import Tensor, auto_convert_to_tensor
    from torchair._ge_concrete_graph import ge_apis as ge
    from torchair._ge_concrete_graph.compat_ir import ge_op, IrDef
    from torchair._ge_concrete_graph.fx2ge_converter import (
        register_fx_node_ge_converter,
    )

    _TORCHAIR_AVAILABLE = True
except ImportError:
    _TORCHAIR_AVAILABLE = False


_FIRST_MX_DTYPE = 2


if _TORCHAIR_AVAILABLE:

    @auto_convert_to_tensor([False, False, False], [False, False, False])
    def AttentionWorkerCombine(
        schedule_context: Tensor,
        expert_scales: Tensor,
        layer_id: Tensor,
        *,
        hidden_size: int,
        token_dtype: int = 0,
        need_schedule: int = 0,
        dependencies=None,
        node_name=None,
    ):
        """REG_OP(AttentionWorkerCombine)
        .INPUT(schedule_context, TensorType({DT_INT8}))
        .INPUT(expert_scales, TensorType({DT_FLOAT, DT_FLOAT8_E8M0}))
        .INPUT(layer_id, TensorType({DT_INT32}))
        .OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16}))
        .OUTPUT(next_layer_id, TensorType({DT_INT32}))
        .REQUIRED_ATTR(hidden_size, Int)
        .ATTR(token_dtype, Int, 0)
        .ATTR(need_schedule, Int, 0)
        """
        return ge_op(
            op_type="AttentionWorkerCombine",
            inputs={
                "schedule_context": schedule_context,
                "expert_scales": expert_scales,
                "layer_id": layer_id,
            },
            attrs={
                "hidden_size": attr.Int(hidden_size),
                "token_dtype": attr.Int(token_dtype),
                "need_schedule": attr.Int(need_schedule),
            },
            outputs=["y", "next_layer_id"],
            dependencies=dependencies,
            node_name=node_name,
            ir=IrDef("AttentionWorkerCombine")
            .input("schedule_context", "DT_INT8")
            .input("expert_scales", "DT_FLOAT, DT_FLOAT8_E8M0")
            .input("layer_id", "DT_INT32")
            .required_attr("hidden_size", attr.Int)
            .attr("token_dtype", attr.Int(0))
            .attr("need_schedule", attr.Int(0))
            .output("y", "DT_FLOAT16, DT_BF16")
            .output("next_layer_id", "DT_INT32"),
        )

    @register_fx_node_ge_converter(
        torch.ops.cann_ops_transformer.attention_worker_combine.default
    )
    def convert_attention_worker_combine(
        schedule_context: Tensor,
        expert_scales: Tensor,
        layer_id: Tensor,
        hidden_size: int,
        *,
        token_dtype: int = 0,
        need_schedule: int = 0,
        meta_outputs=None,
    ):
        """Adapt Torch inputs to GE while preserving raw E8M0 scale encodings."""
        if token_dtype >= _FIRST_MX_DTYPE and expert_scales.dtype == DataType.DT_UINT8:
            expert_scales = ge.Bitcast(expert_scales, type=DataType.DT_FLOAT8_E8M0)
        return AttentionWorkerCombine(
            schedule_context,
            expert_scales,
            layer_id,
            hidden_size=hidden_size,
            token_dtype=token_dtype,
            need_schedule=need_schedule,
        )

else:

    def convert_attention_worker_combine(*args, **kwargs):
        """Report the optional GE dependency without blocking eager imports."""
        raise RuntimeError("AttentionWorkerCombine GE converter requires torchair")
