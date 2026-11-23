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
        [False, False, False, True, True, True, True, True, True],
    )
    def MsaIndexScoreV2(
        query: Tensor,
        key: Tensor,
        start_loc: Tensor,
        block_table: Optional[Tensor],
        query_scale: Optional[Tensor],
        key_scale: Optional[Tensor],
        atten_mask: Optional[Tensor],
        actual_seq_qlen: Optional[Tensor],
        actual_seq_klen: Optional[Tensor],
        *,
        layout_key: str = "BBND",
        sparse_mode: int = 3,
        init_blocks: int = 0,
        local_blocks: int = 1,
        layout_query: str = "TND",
        pool_block_size: int = 128,
        return_choice: int = 0,
        topk: int = 16,
        compress_factor: int = 1,
        dependencies=None,
        node_name=None,
    ):
        if dependencies is None:
            dependencies = []

        inputs = {
            "query": query,
            "key": key,
            "start_loc": start_loc,
            "block_table": block_table,
            "query_scale": query_scale,
            "key_scale": key_scale,
            "atten_mask": atten_mask,
            "actual_seq_qlen": actual_seq_qlen,
            "actual_seq_klen": actual_seq_klen,
        }
        attrs = {
            "layout_key": attr.Str(layout_key),
            "sparse_mode": attr.Int(sparse_mode),
            "init_blocks": attr.Int(init_blocks),
            "local_blocks": attr.Int(local_blocks),
            "layout_query": attr.Str(layout_query),
            "pool_block_size": attr.Int(pool_block_size),
            "return_choice": attr.Int(return_choice),
            "topk": attr.Int(topk),
            "compress_factor": attr.Int(compress_factor),
        }
        outputs = ["score", "sparse_indices", "sparse_values"]
        result = ge_op(
            op_type="MsaIndexScore",
            inputs=inputs,
            attrs=attrs,
            outputs=outputs,
            dependencies=dependencies,
            ir=IrDef("MsaIndexScore")
            .input(
                "query",
                "DT_FLOAT16, DT_BF16, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT4_E2M1",
            )
            .input(
                "key",
                "DT_FLOAT16, DT_BF16, DT_INT8, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT4_E2M1",
            )
            .input("start_loc", "DT_INT32")
            .optional_input("block_table", "DT_INT32")
            .optional_input("query_scale", "DT_FLOAT, DT_FLOAT8_E8M0")
            .optional_input("key_scale", "DT_FLOAT, DT_FLOAT8_E8M0")
            .optional_input("atten_mask", "DT_INT8")
            .optional_input("actual_seq_qlen", "DT_INT32")
            .optional_input("actual_seq_klen", "DT_INT32")
            .attr("layout_key", attr.Str("BBND"))
            .attr("sparse_mode", attr.Int(3))
            .attr("init_blocks", attr.Int(0))
            .attr("local_blocks", attr.Int(1))
            .attr("layout_query", attr.Str("TND"))
            .attr("pool_block_size", attr.Int(128))
            .attr("return_choice", attr.Int(0))
            .attr("topk", attr.Int(16))
            .attr("compress_factor", attr.Int(1))
            .output("score", "DT_FLOAT16, DT_FLOAT")
            .output("sparse_indices", "DT_INT32")
            .output("sparse_values", "DT_FLOAT"),
        )
        if (
            int(return_choice) in (1, 2)
            and isinstance(result, (list, tuple))
            and len(result) > 1
        ):
            if int(return_choice) == 2 and len(result) > 2:
                return result[1], result[2]
            return result[1]
        return result[0] if isinstance(result, (list, tuple)) else result

    @register_fx_node_ge_converter(
        torch.ops.cann_ops_transformer._msa_index_score_v2.default
    )
    def convert_msa_index_score_v2(
        query: Tensor,
        key: Tensor,
        start_loc: Tensor,
        block_table: Optional[Tensor],
        query_scale: Optional[Tensor],
        key_scale: Optional[Tensor],
        atten_mask: Optional[Tensor],
        actual_seq_qlen: Optional[Tensor],
        actual_seq_klen: Optional[Tensor],
        *,
        layout_key: str = "BBND",
        sparse_mode: int = 3,
        init_blocks: int = 0,
        local_blocks: int = 1,
        layout_query: str = "TND",
        pool_block_size: int = 128,
        return_choice: int = 0,
        topk: int = 16,
        compress_factor: int = 1,
        meta_outputs: TensorSpec = None,
    ):
        return MsaIndexScoreV2(
            query=query,
            key=key,
            block_table=block_table,
            key_scale=key_scale,
            atten_mask=atten_mask,
            actual_seq_qlen=actual_seq_qlen,
            actual_seq_klen=actual_seq_klen,
            start_loc=start_loc,
            query_scale=query_scale,
            layout_key=layout_key,
            sparse_mode=sparse_mode,
            init_blocks=init_blocks,
            local_blocks=local_blocks,
            layout_query=layout_query,
            pool_block_size=pool_block_size,
            return_choice=return_choice,
            topk=topk,
            compress_factor=compress_factor,
        )

else:

    def convert_msa_index_score_v2(*args, **kwargs):
        raise RuntimeError(
            "GE converter requires torchair, but torchair is not available."
        )
