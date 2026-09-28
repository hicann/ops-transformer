# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Torch adapter for the migrated npu-ai-operation-kernel LightningIndexer."""

import os
from typing import Optional

import torch
from torch.library import impl

from cann_ops_transformer.op_builder.builder import OpBuilder, get_as_library


class LightningIndexerOpBuilder(OpBuilder):
    def __init__(self):
        super().__init__("lightning_indexer", category="attention")

    def sources(self):
        return ["csrc/attention/lightning_indexer.cpp"]

    def include_paths(self):
        # torch_npu may bundle older ACL headers with the same include guards.
        return [os.path.join(self.cann_path, "include"), *super().include_paths()]

    def schema(self):
        return (
            "lightning_indexer(Tensor query, Tensor key, Tensor weights, *, "
            "Tensor? cur_seq_lengths_query=None, Tensor? cur_seq_lengths_key=None, "
            "Tensor? block_table=None, str layout_query='BSND', str layout_key='PA_BSND', "
            "int sparse_count=2048, int kv_block_len=1, int q_block_len=1, "
            "int init_num=0, int local_num=0, int sparse_mode=3, "
            "int pre_tokens=9223372036854775807, int next_tokens=9223372036854775807, "
            "bool return_value=False) -> (Tensor, Tensor)"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def lightning_indexer_meta(
            query,
            key,
            weights,
            *,
            cur_seq_lengths_query=None,
            cur_seq_lengths_key=None,
            block_table=None,
            layout_query="BSND",
            layout_key="PA_BSND",
            sparse_count=2048,
            kv_block_len=1,
            q_block_len=1,
            init_num=0,
            local_num=0,
            sparse_mode=3,
            pre_tokens=9223372036854775807,
            next_tokens=9223372036854775807,
            return_value=False,
        ):
            if layout_query not in ("BSND", "TND"):
                raise ValueError("layout_query must be BSND or TND")
            if layout_key not in ("BSND", "TND", "PA_BSND"):
                raise ValueError("layout_key must be BSND, TND or PA_BSND")
            if query.dim() != (4 if layout_query == "BSND" else 3):
                raise ValueError("query rank does not match layout_query")
            if key.dim() != (3 if layout_key == "TND" else 4):
                raise ValueError("key rank does not match layout_key")
            if sparse_count <= 0:
                raise ValueError("sparse_count must be positive")
            n2 = key.shape[1 if layout_key == "TND" else 2]
            shape = (*query.shape[:-2], n2, sparse_count)
            return (
                torch.empty(shape, device="meta", dtype=torch.int32),
                torch.empty(shape, device="meta", dtype=query.dtype),
            )


lightning_indexer_op_builder = LightningIndexerOpBuilder()
lightning_indexer_op_builder._ensure_initialized()


@impl(get_as_library(), lightning_indexer_op_builder.name, "PrivateUse1")
def lightning_indexer(
    query: torch.Tensor,
    key: torch.Tensor,
    weights: torch.Tensor,
    *,
    cur_seq_lengths_query: Optional[torch.Tensor] = None,
    cur_seq_lengths_key: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    layout_query: str = "BSND",
    layout_key: str = "PA_BSND",
    sparse_count: int = 2048,
    kv_block_len: int = 1,
    q_block_len: int = 1,
    init_num: int = 0,
    local_num: int = 0,
    sparse_mode: int = 3,
    pre_tokens: int = 9223372036854775807,
    next_tokens: int = 9223372036854775807,
    return_value: bool = False,
):
    module = lightning_indexer_op_builder.load()
    return module.lightning_indexer(
        query,
        key,
        weights,
        cur_seq_lengths_query,
        cur_seq_lengths_key,
        block_table,
        layout_query,
        layout_key,
        sparse_count,
        kv_block_len,
        q_block_len,
        init_num,
        local_num,
        sparse_mode,
        pre_tokens,
        next_tokens,
        return_value,
    )
