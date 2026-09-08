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
    from torchair._ge_concrete_graph.fx2ge_converter import (
        register_fx_node_ge_converter,
    )
    from torchair.ge._ge_graph import Tensor, TensorSpec
    from torchair.ge import attr
    from typing import Optional, List

    _TORCHAIR_AVAILABLE = True
except ImportError:
    _TORCHAIR_AVAILABLE = False

if _TORCHAIR_AVAILABLE:

    def _pool_key_stride0(pool_key) -> int:
        """Read stride(0) of pool_key at conversion time (element unit).

        Used for PA_BBND 0-axis non-contiguous addressing: the eager/aclnn path
        reads at::Tensor::stride(0) in the C++ wrapper (csrc/pool_key_indexer.cpp)
        and passes it as the key_stride0 attribute; the GE graph path must do the
        same because the host tiling consumes the attribute first (see
        pool_key_indexer_tiling.cpp, priority: attr > runtime stride > shape).

        Returns -1 (not specified) when stride metadata is unavailable at
        conversion time (e.g. contiguous input or older torchair); the host
        tiling then falls back to the runtime stride reported by the framework
        or the shape-derived contiguous stride. A specified value conflicting
        with the runtime stride is rejected by the host tiling.
        """
        for accessor in (
            lambda t: t.stride(0),  # torchair ge Tensor with stride metadata
            lambda t: t.stride[0],  # stride exposed as a plain sequence
            lambda t: t.tensor.stride(0),  # wrapped original torch tensor
        ):
            try:
                val = accessor(pool_key)
                if callable(val):
                    val = val()
                return int(val)
            except (AttributeError, TypeError, ValueError, IndexError):
                continue
        return -1

    @register_fx_node_ge_converter(
        torch.ops.cann_ops_transformer.pool_key_indexer.default
    )
    def convert_pool_key_indexer(
        query: Tensor,
        pool_key: Tensor,
        weights: Tensor,
        pool_tail_k: Tensor,
        *,
        actual_seq_q: Optional[Tensor] = None,
        actual_seq_k: Optional[Tensor] = None,
        block_table: Optional[Tensor] = None,
        q_descale: Optional[Tensor] = None,
        k_descale: Optional[Tensor] = None,
        layout_q: str = "BSND",
        layout_k: str = "BSND",
        topk: int = 2048,
        pool_size: int = 16,
        mask_mode: int = 3,
        quant_mode: int = -1,
        return_value: bool = False,
        meta_outputs: List[TensorSpec] = None,
    ):
        return torchair.ge.custom_op(
            "PoolKeyIndexer",
            inputs={
                "query": query,
                "pool_key": pool_key,
                "weights": weights,
                "pool_tail_k": pool_tail_k,
                "actual_seq_q": actual_seq_q,
                "actual_seq_k": actual_seq_k,
                "block_table": block_table,
                "q_descale": q_descale,
                "k_descale": k_descale,
            },
            attrs={
                "layout_q": attr.Str(layout_q),
                "layout_k": attr.Str(layout_k),
                "topk": attr.Int(topk),
                "pool_size": attr.Int(pool_size),
                "mask_mode": attr.Int(mask_mode),
                "quant_mode": attr.Int(quant_mode),
                "return_value": attr.Bool(return_value),
                # PA_BBND 0-axis non-contiguous support (attr index 7/8, must be
                # the last attrs to match the op_def registration order)
                "key_stride0": attr.Int(_pool_key_stride0(pool_key)),
                "k_descale_stride0": attr.Int(_pool_key_stride0(k_descale)),
            },
            outputs=["sparse_indices_out", "sparse_values_out"],
        )
