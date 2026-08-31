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
    from torchair.ge._ge_graph import DataType, TensorType
    from torchair.ge._ge_graph import compat_as_bytes, compat_as_bytes_list
    from torchair.ge._ge_graph import trans_to_list_list_int, trans_to_list_list_float
    from torchair.ge._ge_graph import get_invalid_desc
    from torchair._ge_concrete_graph.compat_ir import ge_op, IrDef
    from torchair.ge import attr

    _TORCHAIR_AVAILABLE = True
except ImportError:
    _TORCHAIR_AVAILABLE = False

INT64_MAX = 9223372036854775807

if _TORCHAIR_AVAILABLE:

    @auto_convert_to_tensor(
        [False, False, False, False, False, False, False, False, False, False],
        [False, False, False, False, False, False, False, False, False, False],
    )
    def KvQuantSparseFlashAttention(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        sparse_indices: Tensor,
        key_dequant_scale: Optional[Tensor],
        value_dequant_scale: Optional[Tensor],
        block_table: Optional[Tensor],
        actual_seq_lengths_query: Optional[Tensor],
        actual_seq_lengths_kv: Optional[Tensor],
        sinks: Optional[Tensor],
        *,
        scale_value: float = 1.0,
        key_quant_mode: int = 1,
        value_quant_mode: int = 1,
        sparse_block_size: int = 1,
        layout_query: str = "BSND",
        layout_kv: str = "BSND",
        sparse_mode: int = 3,
        pre_tokens: int = INT64_MAX,
        next_tokens: int = INT64_MAX,
        attention_mode: int = 0,
        quant_scale_repo_mode: int = 1,
        tile_size: int = 128,
        rope_head_dim: int = 64,
        key_dtype: Optional[int] = None,
        value_dtype: Optional[int] = None,
        dependencies=[],
        node_name=None,
    ):
        """REG_OP(KvQuantSparseFlashAttentionV2)\n
        .INPUT(query, TensorType({DT_FLOAT16, DT_BF16}))\n
        .INPUT(key, TensorType({DT_FLOAT8_E4M3FN, DT_HIFLOAT8, DT_INT8}))\n
        .INPUT(value, TensorType({DT_FLOAT8_E4M3FN, DT_HIFLOAT8, DT_INT8}))\n
        .INPUT(sparse_indices, TensorType({DT_INT32}))\n
        .OPTIONAL_INPUT(key_dequant_scale, TensorType({DT_FLOAT}))\n
        .OPTIONAL_INPUT(value_dequant_scale, TensorType({DT_FLOAT}))\n
        .OPTIONAL_INPUT(block_table, TensorType({DT_INT32}))\n
        .OPTIONAL_INPUT(actual_seq_lengths_query, TensorType({DT_INT32}))\n
        .OPTIONAL_INPUT(actual_seq_lengths_kv, TensorType({DT_INT32}))\n
        .OPTIONAL_INPUT(sinks, TensorType({DT_FLOAT}))\n
        .OUTPUT(attention_out, TensorType({DT_FLOAT16, DT_BF16}))\n
        .ATTR(scale_value, Float, 1.0)\n
        .ATTR(key_quant_mode, Int, 1)\n
        .ATTR(value_quant_mode, Int, 1)\n
        .ATTR(sparse_block_size, Int, 1)\n
        .ATTR(layout_query, String, "BSND")\n
        .ATTR(layout_kv, String, "BSND")\n
        .ATTR(sparse_mode, Int, 3)\n
        .ATTR(pre_tokens, Int, 9223372036854775807)\n
        .ATTR(next_tokens, Int, 9223372036854775807)\n
        .ATTR(attention_mode, Int, 0)\n
        .ATTR(quant_scale_repo_mode, Int, 1)\n
        .ATTR(tile_size, Int, 128)\n
        .ATTR(rope_head_dim, Int, 64)\n
        """
        inputs = {
            "query": query,
            "key": key,
            "value": value,
            "sparse_indices": sparse_indices,
            "key_dequant_scale": key_dequant_scale,
            "value_dequant_scale": value_dequant_scale,
            "block_table": block_table,
            "actual_seq_lengths_query": actual_seq_lengths_query,
            "actual_seq_lengths_kv": actual_seq_lengths_kv,
            "sinks": sinks,
        }

        attrs = {
            "scale_value": attr.Float(scale_value),
            "key_quant_mode": attr.Int(key_quant_mode),
            "value_quant_mode": attr.Int(value_quant_mode),
            "sparse_block_size": attr.Int(sparse_block_size),
            "layout_query": attr.Str(layout_query),
            "layout_kv": attr.Str(layout_kv),
            "sparse_mode": attr.Int(sparse_mode),
            "pre_tokens": attr.Int(pre_tokens),
            "next_tokens": attr.Int(next_tokens),
            "attention_mode": attr.Int(attention_mode),
            "quant_scale_repo_mode": attr.Int(quant_scale_repo_mode),
            "tile_size": attr.Int(tile_size),
            "rope_head_dim": attr.Int(rope_head_dim),
        }

        outputs = [
            "attention_out",
        ]

        return ge_op(
            op_type="KvQuantSparseFlashAttentionV2",
            inputs=inputs,
            attrs=attrs,
            outputs=outputs,
            dependencies=dependencies,
            ir=IrDef("KvQuantSparseFlashAttentionV2")
            .input("query", "DT_FLOAT16, DT_BF16")
            .input("key", "DT_FLOAT8_E4M3FN, DT_HIFLOAT8, DT_INT8")
            .input("value", "DT_FLOAT8_E4M3FN, DT_HIFLOAT8, DT_INT8")
            .input("sparse_indices", "DT_INT32")
            .optional_input("key_dequant_scale", "DT_FLOAT")
            .optional_input("value_dequant_scale", "DT_FLOAT")
            .optional_input("block_table", "DT_INT32")
            .optional_input("actual_seq_lengths_query", "DT_INT32")
            .optional_input("actual_seq_lengths_kv", "DT_INT32")
            .optional_input("sinks", "DT_FLOAT")
            .attr("scale_value", attr.Float(1.0))
            .attr("key_quant_mode", attr.Int(1))
            .attr("value_quant_mode", attr.Int(1))
            .attr("sparse_block_size", attr.Int(1))
            .attr("layout_query", attr.Str("BSND"))
            .attr("layout_kv", attr.Str("BSND"))
            .attr("sparse_mode", attr.Int(3))
            .attr("pre_tokens", attr.Int(INT64_MAX))
            .attr("next_tokens", attr.Int(INT64_MAX))
            .attr("attention_mode", attr.Int(0))
            .attr("quant_scale_repo_mode", attr.Int(1))
            .attr("tile_size", attr.Int(128))
            .attr("rope_head_dim", attr.Int(64))
            .output("attention_out", "DT_FLOAT16, DT_BF16"),
        )

    @register_fx_node_ge_converter(
        torch.ops.cann_ops_transformer.kv_quant_sparse_flash_attention.default
    )
    def convert_kv_quant_sparse_flash_attention(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        sparse_indices: Tensor,
        scale_value: float = 1.0,
        key_quant_mode: int = 1,
        value_quant_mode: int = 1,
        key_dequant_scale: Tensor = None,
        value_dequant_scale: Tensor = None,
        block_table: Tensor = None,
        actual_seq_lengths_query: Tensor = None,
        actual_seq_lengths_kv: Tensor = None,
        sparse_block_size: int = 1,
        layout_query: str = "BSND",
        layout_kv: str = "BSND",
        sparse_mode: int = 3,
        pre_tokens: int = INT64_MAX,
        next_tokens: int = INT64_MAX,
        attention_mode: int = 0,
        quant_scale_repo_mode: int = 1,
        tile_size: int = 128,
        rope_head_dim: int = 64,
        key_dtype: Optional[int] = None,
        value_dtype: Optional[int] = None,
        sinks: Tensor = None,
        meta_outputs: TensorSpec = None,
    ):
        return KvQuantSparseFlashAttention(
            query=query,
            key=key,
            value=value,
            sparse_indices=sparse_indices,
            key_dequant_scale=key_dequant_scale,
            value_dequant_scale=value_dequant_scale,
            block_table=block_table,
            actual_seq_lengths_query=actual_seq_lengths_query,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            sinks=sinks,
            scale_value=scale_value,
            key_quant_mode=key_quant_mode,
            value_quant_mode=value_quant_mode,
            sparse_block_size=sparse_block_size,
            layout_query=layout_query,
            layout_kv=layout_kv,
            sparse_mode=sparse_mode,
            pre_tokens=pre_tokens,
            next_tokens=next_tokens,
            attention_mode=attention_mode,
            quant_scale_repo_mode=quant_scale_repo_mode,
            tile_size=tile_size,
            rope_head_dim=rope_head_dim,
            key_dtype=key_dtype,
            value_dtype=value_dtype,
        )
