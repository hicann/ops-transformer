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
        register_fx_node_ge_converter,
    )
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

if _TORCHAIR_AVAILABLE:

    @auto_convert_to_tensor([False, False, False, False], [False, False, False, False])
    def EngramHash(
        ngram_token_ids: Tensor,
        multipliers: Tensor,
        vocab_sizes: Tensor,
        offsets: Tensor,
        *,
        dependencies=[],
        node_name=None,
    ):
        """REG_OP(EngramHash)\n
        .INPUT(ngram_token_ids, TensorType({DT_INT32}))\n
        .INPUT(multipliers, TensorType({DT_INT64}))\n
        .INPUT(vocab_sizes, TensorType({DT_INT32}))\n
        .INPUT(offsets, TensorType({DT_INT32}))\n
        .OUTPUT(hash_indices, TensorType({DT_INT32}))\n
        """
        inputs = {
            "ngram_token_ids": ngram_token_ids,
            "multipliers": multipliers,
            "vocab_sizes": vocab_sizes,
            "offsets": offsets,
        }

        outputs = [
            "hash_indices",
        ]

        return ge_op(
            op_type="EngramHash",
            inputs=inputs,
            attrs={},
            outputs=outputs,
            dependencies=dependencies,
            ir=IrDef("EngramHash")
            .input("ngram_token_ids", "DT_INT32")
            .input("multipliers", "DT_INT64")
            .input("vocab_sizes", "DT_INT32")
            .input("offsets", "DT_INT32")
            .output("hash_indices", "DT_INT32"),
        )

    @register_fx_node_ge_converter(torch.ops.cann_ops_transformer.engram_hash.default)
    def convert_engram_hash(
        ngram_token_ids: Tensor,
        multipliers: Tensor,
        vocab_sizes: Tensor,
        offsets: Tensor,
        meta_outputs: TensorSpec = None,
    ):
        return EngramHash(
            ngram_token_ids=ngram_token_ids,
            multipliers=multipliers,
            vocab_sizes=vocab_sizes,
            offsets=offsets,
        )
