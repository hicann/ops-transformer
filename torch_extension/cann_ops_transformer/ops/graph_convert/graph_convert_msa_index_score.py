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
# Atlas A2/A3 only; key layouts: PA BBND/BNBD and TND packed. Does not support Ascend 950 / FP8.

try:
    import torch
    import torch_npu
    import torchair
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
        [False, False, False, False, False, False, False, False],
        [False, False, True, True, True, True, True, False],
    )
    def MsaIndexScore(
        query: Tensor,
        key: Tensor,
        block_table: Optional[Tensor],
        scale: Optional[Tensor],
        atten_mask: Optional[Tensor],
        actual_seq_qlen: Optional[Tensor],
        actual_seq_klen: Optional[Tensor],
        start_loc: Tensor,
        *,
        layout_key: str = "BBND",
        sparse_mode: int = 3,
        init_blocks: int = 0,
        local_blocks: int = 1,
        dependencies=None,
        node_name=None,
    ):
        if dependencies is None:
            dependencies = []

        inputs = {
            "query": query,
            "key": key,
            "block_table": block_table,
            "scale": scale,
            "atten_mask": atten_mask,
            "actual_seq_qlen": actual_seq_qlen,
            "actual_seq_klen": actual_seq_klen,
            "start_loc": start_loc,
        }
        attrs = {
            "layout_key": attr.Str(layout_key),
            "sparse_mode": attr.Int(sparse_mode),
            "init_blocks": attr.Int(init_blocks),
            "local_blocks": attr.Int(local_blocks),
        }
        outputs = ["score"]
        return ge_op(
            op_type="MsaIndexScore",
            inputs=inputs,
            attrs=attrs,
            outputs=outputs,
            dependencies=dependencies,
            ir=IrDef("MsaIndexScore")
            .input("query", "DT_FLOAT16, DT_BF16")
            .input("key", "DT_FLOAT16, DT_BF16, DT_INT8")
            .optional_input("block_table", "DT_INT32")
            .optional_input("scale", "DT_FLOAT")
            .optional_input("atten_mask", "DT_INT8")
            .optional_input("actual_seq_qlen", "DT_INT32")
            .optional_input("actual_seq_klen", "DT_INT32")
            .input("start_loc", "DT_INT32")
            .attr("layout_key", attr.Str("BBND"))
            .attr("sparse_mode", attr.Int(3))
            .attr("init_blocks", attr.Int(0))
            .attr("local_blocks", attr.Int(1))
            .output("score", "DT_FLOAT"),
        )

    @register_fx_node_ge_converter(
        torch.ops.cann_ops_transformer.msa_index_score.default
    )
    def convert_msa_index_score(
        query: Tensor,
        key: Tensor,
        block_table: Optional[Tensor],
        scale: Optional[Tensor],
        atten_mask: Optional[Tensor],
        actual_seq_qlen: Optional[Tensor],
        actual_seq_klen: Optional[Tensor],
        start_loc: Tensor,
        *,
        layout_key: str = "BBND",
        sparse_mode: int = 3,
        init_blocks: int = 0,
        local_blocks: int = 1,
        meta_outputs: TensorSpec = None,
    ):
        return MsaIndexScore(
            query=query,
            key=key,
            block_table=block_table,
            scale=scale,
            atten_mask=atten_mask,
            actual_seq_qlen=actual_seq_qlen,
            actual_seq_klen=actual_seq_klen,
            start_loc=start_loc,
            layout_key=layout_key,
            sparse_mode=sparse_mode,
            init_blocks=init_blocks,
            local_blocks=local_blocks,
        )

else:

    def convert_msa_index_score(*args, **kwargs):
        raise RuntimeError(
            "GE converter requires torchair, but torchair is not available."
        )
