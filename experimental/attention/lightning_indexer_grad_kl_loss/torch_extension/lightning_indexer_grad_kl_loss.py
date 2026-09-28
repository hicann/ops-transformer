# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Torch interfaces adapted from LightningIndexerGradKlLossNpuOpApi.cpp."""

import os

import torch
import torch_npu
from torch.library import impl

from cann_ops_transformer.op_builder.builder import OpBuilder, get_as_library


class LightningIndexerGradKlLossOpBuilder(OpBuilder):
    def __init__(self):
        super().__init__("lightning_indexer_grad_kl_loss", category="attention")

    def sources(self):
        return ["csrc/attention/lightning_indexer_grad_kl_loss.cpp"]

    def include_paths(self):
        return [os.path.join(self.cann_path, "include"), *super().include_paths()]

    def schema(self):
        inputs = (
            "Tensor query, Tensor key, Tensor query_index, Tensor key_index, "
            "Tensor weights, Tensor sparse_indices, Tensor softmax_max, Tensor softmax_sum, "
            "Tensor? query_rope, Tensor? key_rope, "
        )
        attrs = (
            "int[]? cur_seq_lengths_query=None, int[]? cur_seq_lengths_key=None, "
            "float scale_value=1., str? layout='BSND', int? sparse_mode=3, "
            "int? pre_tokens=2147483647, int? next_tokens=2147483647, "
            "int block_size=1, bool? deterministic=False"
        )
        returns = " -> (Tensor, Tensor, Tensor, Tensor)"
        return [
            f"lightning_indexer_grad_kl_loss({inputs}{attrs}){returns}",
            f"lightning_indexer_grad_kl_loss_skip_padding({inputs}Tensor? mask, "
            f"{attrs}, int? validTokenNum=-1){returns}",
        ]

    def register_meta(self):
        @impl(get_as_library(), "lightning_indexer_grad_kl_loss", "Meta")
        def grad_meta(
            query,
            key,
            query_index,
            key_index,
            weights,
            sparse_indices,
            softmax_max,
            softmax_sum,
            query_rope,
            key_rope,
            cur_seq_lengths_query=None,
            cur_seq_lengths_key=None,
            scale_value=1.0,
            layout="BSND",
            sparse_mode=3,
            pre_tokens=2147483647,
            next_tokens=2147483647,
            block_size=1,
            deterministic=False,
        ):
            return (
                torch.empty_like(query_index, device="meta"),
                torch.empty_like(key_index, device="meta"),
                torch.empty_like(weights, device="meta"),
                torch.empty((1,), dtype=torch.float32, device="meta"),
            )

        @impl(get_as_library(), "lightning_indexer_grad_kl_loss_skip_padding", "Meta")
        def grad_skip_meta(
            query,
            key,
            query_index,
            key_index,
            weights,
            sparse_indices,
            softmax_max,
            softmax_sum,
            query_rope,
            key_rope,
            mask,
            cur_seq_lengths_query=None,
            cur_seq_lengths_key=None,
            scale_value=1.0,
            layout="BSND",
            sparse_mode=3,
            pre_tokens=2147483647,
            next_tokens=2147483647,
            block_size=1,
            deterministic=False,
            validTokenNum=-1,
        ):
            return grad_meta(
                query,
                key,
                query_index,
                key_index,
                weights,
                sparse_indices,
                softmax_max,
                softmax_sum,
                query_rope,
                key_rope,
            )


def _format_inputs(*tensors):
    # Use the registered public API: bundled C++ headers and libtorch_npu can
    # expose different private npu_format_cast signatures.
    return tuple(
        torch_npu.npu_format_cast(t, 2) if t is not None else None for t in tensors
    )


lightning_indexer_grad_kl_loss_op_builder = LightningIndexerGradKlLossOpBuilder()
lightning_indexer_grad_kl_loss_op_builder._ensure_initialized()


@impl(get_as_library(), "lightning_indexer_grad_kl_loss", "PrivateUse1")
def lightning_indexer_grad_kl_loss(
    query,
    key,
    query_index,
    key_index,
    weights,
    sparse_indices,
    softmax_max,
    softmax_sum,
    query_rope,
    key_rope,
    cur_seq_lengths_query=None,
    cur_seq_lengths_key=None,
    scale_value=1.0,
    layout="BSND",
    sparse_mode=3,
    pre_tokens=2147483647,
    next_tokens=2147483647,
    block_size=1,
    deterministic=False,
):
    (
        query,
        key,
        query_index,
        key_index,
        weights,
        sparse_indices,
        softmax_max,
        softmax_sum,
        query_rope,
        key_rope,
    ) = _format_inputs(
        query,
        key,
        query_index,
        key_index,
        weights,
        sparse_indices,
        softmax_max,
        softmax_sum,
        query_rope,
        key_rope,
    )
    module = lightning_indexer_grad_kl_loss_op_builder.load()
    return module.lightning_indexer_grad_kl_loss(
        query,
        key,
        query_index,
        key_index,
        weights,
        sparse_indices,
        softmax_max,
        softmax_sum,
        query_rope,
        key_rope,
        cur_seq_lengths_query,
        cur_seq_lengths_key,
        scale_value,
        layout,
        sparse_mode,
        pre_tokens,
        next_tokens,
        block_size,
        deterministic,
    )


@impl(get_as_library(), "lightning_indexer_grad_kl_loss_skip_padding", "PrivateUse1")
def lightning_indexer_grad_kl_loss_skip_padding(
    query,
    key,
    query_index,
    key_index,
    weights,
    sparse_indices,
    softmax_max,
    softmax_sum,
    query_rope,
    key_rope,
    mask,
    cur_seq_lengths_query=None,
    cur_seq_lengths_key=None,
    scale_value=1.0,
    layout="BSND",
    sparse_mode=3,
    pre_tokens=2147483647,
    next_tokens=2147483647,
    block_size=1,
    deterministic=False,
    validTokenNum=-1,
):
    (
        query,
        key,
        query_index,
        key_index,
        weights,
        sparse_indices,
        softmax_max,
        softmax_sum,
        query_rope,
        key_rope,
        mask,
    ) = _format_inputs(
        query,
        key,
        query_index,
        key_index,
        weights,
        sparse_indices,
        softmax_max,
        softmax_sum,
        query_rope,
        key_rope,
        mask,
    )
    module = lightning_indexer_grad_kl_loss_op_builder.load()
    return module.lightning_indexer_grad_kl_loss_skip_padding(
        query,
        key,
        query_index,
        key_index,
        weights,
        sparse_indices,
        softmax_max,
        softmax_sum,
        query_rope,
        key_rope,
        mask,
        cur_seq_lengths_query,
        cur_seq_lengths_key,
        scale_value,
        layout,
        sparse_mode,
        pre_tokens,
        next_tokens,
        block_size,
        deterministic,
        validTokenNum,
    )
