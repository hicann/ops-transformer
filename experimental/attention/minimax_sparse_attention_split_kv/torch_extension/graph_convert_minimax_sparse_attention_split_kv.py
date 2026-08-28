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
    from typing import Optional
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

if _TORCHAIR_AVAILABLE:

    @auto_convert_to_tensor(
        [False, False, False, False, False, False, False, False, False],
        [False, False, False, True, False, False, False, False, False],
    )
    def MinimaxSparseAttentionSplitKv(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        blockTable: Optional[Tensor],
        k2qRowPtr: Tensor,
        k2qQIndices: Tensor,
        k2qSlotIndices: Tensor,
        actualSeqLengths: Tensor,
        actualSeqLengthsKv: Tensor,
        *,
        numKeyValueHeads: int = 1,
        scaleValue: float = 0.0,
        blockSize: int = 128,
        topK: int = 8,
        innerPrecise: int = 4,
        softmaxLseFlag: bool = False,
        inputLayout: str = "TND",
        dependencies=None,
        node_name=None,
    ):
        """REG_OP(MinimaxSparseAttentionSplitKv)
        .INPUT(query, TensorType({DT_BF16}))
        .INPUT(key, TensorType({DT_BF16}))
        .INPUT(value, TensorType({DT_BF16}))
        .OPTIONAL_INPUT(blockTable, TensorType({DT_INT32}))
        .INPUT(k2qRowPtr, TensorType({DT_INT32}))
        .INPUT(k2qQIndices, TensorType({DT_INT32}))
        .INPUT(k2qSlotIndices, TensorType({DT_INT32}))
        .INPUT(actualSeqLengths, TensorType({DT_INT32}))
        .INPUT(actualSeqLengthsKv, TensorType({DT_INT32}))
        .OUTPUT(attentionOut, TensorType({DT_BF16}))
        .OUTPUT(softmaxLse, TensorType({DT_FLOAT}))
        .ATTR(numKeyValueHeads, Int, 1)
        .ATTR(scaleValue, Float, 0.0)
        .ATTR(blockSize, Int, 128)
        .ATTR(topK, Int, 8)
        .ATTR(innerPrecise, Int, 4)
        .ATTR(softmaxLseFlag, Bool, false)
        .ATTR(inputLayout, String, "TND")
        """
        if dependencies is None:
            dependencies = []

        inputs = {
            "query": query,
            "key": key,
            "value": value,
            "blockTable": blockTable,
            "k2qRowPtr": k2qRowPtr,
            "k2qQIndices": k2qQIndices,
            "k2qSlotIndices": k2qSlotIndices,
            "actualSeqLengths": actualSeqLengths,
            "actualSeqLengthsKv": actualSeqLengthsKv,
        }
        attrs = {
            "numKeyValueHeads": attr.Int(numKeyValueHeads),
            "scaleValue": attr.Float(scaleValue),
            "blockSize": attr.Int(blockSize),
            "topK": attr.Int(topK),
            "innerPrecise": attr.Int(innerPrecise),
            "softmaxLseFlag": attr.Bool(softmaxLseFlag),
            "inputLayout": attr.Str(inputLayout),
        }
        outputs = ["attentionOut", "softmaxLse"]
        return ge_op(
            op_type="MinimaxSparseAttentionSplitKv",
            inputs=inputs,
            attrs=attrs,
            outputs=outputs,
            dependencies=dependencies,
            node_name=node_name,
            ir=IrDef("MinimaxSparseAttentionSplitKv")
            .input("query", "DT_BF16")
            .input("key", "DT_BF16")
            .input("value", "DT_BF16")
            .optional_input("blockTable", "DT_INT32")
            .input("k2qRowPtr", "DT_INT32")
            .input("k2qQIndices", "DT_INT32")
            .input("k2qSlotIndices", "DT_INT32")
            .input("actualSeqLengths", "DT_INT32")
            .input("actualSeqLengthsKv", "DT_INT32")
            .attr("numKeyValueHeads", attr.Int(1))
            .attr("scaleValue", attr.Float(0.0))
            .attr("blockSize", attr.Int(128))
            .attr("topK", attr.Int(8))
            .attr("innerPrecise", attr.Int(4))
            .attr("softmaxLseFlag", attr.Bool(False))
            .attr("inputLayout", attr.Str("TND"))
            .output("attentionOut", "DT_BF16")
            .output("softmaxLse", "DT_FLOAT"),
        )

    @register_fx_node_ge_converter(
        torch.ops.cann_ops_transformer.minimax_sparse_attention_split_kv.default
    )
    def convert_minimax_sparse_attention_split_kv(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        block_table: Optional[Tensor],
        k2q_row_ptr: Tensor,
        k2q_q_indices: Tensor,
        k2q_slot_indices: Tensor,
        actual_seq_lengths: Tensor,
        actual_seq_lengths_kv: Tensor,
        num_key_value_heads: int,
        scale_value: float,
        block_size: int,
        top_k: int,
        inner_precise: int = 4,
        softmax_lse_flag: bool = False,
        input_layout: str = "TND",
        meta_outputs: TensorSpec = None,
    ):
        return MinimaxSparseAttentionSplitKv(
            query=query,
            key=key,
            value=value,
            blockTable=block_table,
            k2qRowPtr=k2q_row_ptr,
            k2qQIndices=k2q_q_indices,
            k2qSlotIndices=k2q_slot_indices,
            actualSeqLengths=actual_seq_lengths,
            actualSeqLengthsKv=actual_seq_lengths_kv,
            numKeyValueHeads=num_key_value_heads,
            scaleValue=scale_value,
            blockSize=block_size,
            topK=top_k,
            innerPrecise=inner_precise,
            softmaxLseFlag=softmax_lse_flag,
            inputLayout=input_layout,
        )

else:

    def convert_minimax_sparse_attention_split_kv(*args, **kwargs):
        raise RuntimeError(
            "GE converter requires torchair, but torchair is not available."
        )
