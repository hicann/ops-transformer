# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from typing import Optional
import torch
from torch.library import impl
from cann_ops_transformer.op_builder import OpBuilder, get_as_library


class PoolKeyIndexerOpBuilder(OpBuilder):
    def __init__(self):
        super(PoolKeyIndexerOpBuilder, self).__init__(
            "pool_key_indexer", category="attention"
        )

    def sources(self):
        """Path to C++ source code."""
        return ["csrc/attention/pool_key_indexer.cpp"]

    def schema(self) -> str:
        """PyTorch operator signature.

        Required tensors + required scalars first, then optional tensors after '*'.
        Attr defaults align with op_host/pool_key_indexer_def.cpp.
        """
        return (
            "pool_key_indexer(Tensor query, Tensor pool_key, Tensor weights, "
            "Tensor pool_tail_k, *, "
            "Tensor? actual_seq_q=None, Tensor? actual_seq_k=None, "
            "Tensor? block_table=None, Tensor? q_descale=None, Tensor? k_descale=None, "
            'str layout_q="BSND", str layout_k="BSND", '
            "int topk=2048, int pool_size=16, int mask_mode=3, int quant_mode=-1, "
            "bool return_value=False) -> (Tensor, Tensor)"
        )

    def register_meta(self):
        """
        Registers the Meta implementation (Shape/Dtype inference).
        Essential for Autograd and FakeTensor support.
        """

        @impl(get_as_library(), self.name, "Meta")
        def pool_key_indexer_meta(
            query,
            pool_key,
            weights,
            pool_tail_k,
            *,
            actual_seq_q=None,
            actual_seq_k=None,
            block_table=None,
            q_descale=None,
            k_descale=None,
            layout_q="BSND",
            layout_k="BSND",
            topk=2048,
            pool_size=16,
            mask_mode=3,
            quant_mode=-1,
            return_value=False,
        ):
            indices_len = topk + pool_size - 1
            values_len = topk // pool_size
            if layout_q == "BSND":
                idx_shape = [query.shape[0], query.shape[1], indices_len]
                val_shape = [query.shape[0], query.shape[1], values_len]
            else:
                idx_shape = [query.shape[0], indices_len]
                val_shape = [query.shape[0], values_len]
            sparse_indices = torch.empty(idx_shape, dtype=torch.int32, device="meta")
            if return_value:
                sparse_values = torch.empty(
                    val_shape, dtype=torch.float32, device="meta"
                )
            else:
                sparse_values = torch.empty([0], dtype=torch.float32, device="meta")
            return (sparse_indices, sparse_values)


# Instantiate the builder (this registers schema + meta)
pool_key_indexer_op_builder = PoolKeyIndexerOpBuilder()
pool_key_indexer_op_builder._ensure_initialized()


@impl(get_as_library(), pool_key_indexer_op_builder.name, "PrivateUse1")
def pool_key_indexer(
    query,
    pool_key,
    weights,
    pool_tail_k,
    *,
    actual_seq_q=None,
    actual_seq_k=None,
    block_table=None,
    q_descale=None,
    k_descale=None,
    layout_q="BSND",
    layout_k="BSND",
    topk=2048,
    pool_size=16,
    mask_mode=3,
    quant_mode=-1,
    return_value=False,
):
    """
    Dispatcher implementation for NPU.
    'PrivateUse1' is the dispatch key for custom NPU backends.
    """
    op_module = pool_key_indexer_op_builder.load()
    return op_module.pool_key_indexer(
        query,
        pool_key,
        weights,
        pool_tail_k,
        actual_seq_q,
        actual_seq_k,
        block_table,
        q_descale,
        k_descale,
        layout_q,
        layout_k,
        topk,
        pool_size,
        mask_mode,
        quant_mode,
        return_value,
    )
