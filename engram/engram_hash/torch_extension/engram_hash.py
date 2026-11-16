# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import torch
from torch.library import impl
from cann_ops_transformer.op_builder import OpBuilder, get_as_library


class EngramHashOpBuilder(OpBuilder):
    """EngramHash 的 PyTorch 扩展构建器。

    输入范围约束:
      - ngram_token_ids: INT32 [num_tokens, max_ngram_size] (N >= 2, num_tokens 可为 0);
      - multipliers: INT64 [num_ngram_layers, max_ngram_size];
      - vocab_sizes: INT32 [num_ngram_layers, max_ngram_size - 1, num_embed_table_per_ngram];
      - offsets: INT32 [num_ngram_layers, (max_ngram_size - 1) * num_embed_table_per_ngram];
      - 输出为 INT32 [num_ngram_layers, num_tokens, num_out_cols] 的 hash 索引。
    """

    def __init__(self):
        super(EngramHashOpBuilder, self).__init__("engram_hash", category="engram")

    def sources(self):
        """Path to C++ source code."""
        return ["csrc/engram/engram_hash.cpp"]

    def schema(self) -> str:
        """PyTorch operator signature."""
        return "engram_hash(Tensor ngram_token_ids, Tensor multipliers, Tensor vocab_sizes, Tensor offsets) -> Tensor"

    def register_meta(self):
        """
        Registers the Meta implementation.
        """

        @impl(get_as_library(), self.name, "Meta")
        def engram_hash_meta(
            ngram_token_ids,
            multipliers,
            vocab_sizes,
            offsets,
        ) -> torch.Tensor:
            num_ngram_layers = vocab_sizes.shape[0]
            num_tokens = ngram_token_ids.shape[0]
            num_out_cols = (ngram_token_ids.shape[1] - 1) * vocab_sizes.shape[2]
            return torch.empty(
                (num_ngram_layers, num_tokens, num_out_cols),
                dtype=torch.int32,
                device=ngram_token_ids.device,
            )


# Instantiate the builder
engram_hash_op_builder = EngramHashOpBuilder()
engram_hash_op_builder._ensure_initialized()


@impl(get_as_library(), engram_hash_op_builder.name, "PrivateUse1")
def _engram_hash(
    ngram_token_ids: torch.Tensor,
    multipliers: torch.Tensor,
    vocab_sizes: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    op_module = engram_hash_op_builder.load()
    return op_module.engram_hash(ngram_token_ids, multipliers, vocab_sizes, offsets)


def engram_hash(
    ngram_token_ids: torch.Tensor,
    multipliers: torch.Tensor,
    vocab_sizes: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    """计算 Engram n-gram hash 的 embedding 索引。

    ``ngram_token_ids`` 为 INT32 ``(num_tokens, max_ngram_size)``，
    ``multipliers`` 为 INT64，``vocab_sizes`` / ``offsets`` 为 INT32。
    返回 INT32 ``(num_ngram_layers, num_tokens, num_out_cols)`` 的索引张量。
    """
    return torch.ops.cann_ops_transformer.engram_hash(
        ngram_token_ids,
        multipliers,
        vocab_sizes,
        offsets,
    )
