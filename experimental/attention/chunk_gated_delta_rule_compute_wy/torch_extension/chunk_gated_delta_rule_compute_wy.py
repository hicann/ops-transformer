# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from typing import Tuple

import torch
from torch.library import impl

from cann_ops_transformer.op_builder import OpBuilder, get_as_library


class ChunkGatedDeltaRuleComputeWyOpBuilder(OpBuilder):
    def __init__(self):
        super().__init__("chunk_gated_delta_rule_compute_wy", category="attention")

    def sources(self):
        return ["csrc/attention/chunk_gated_delta_rule_compute_wy.cpp"]

    def schema(self) -> str:
        return (
            "chunk_gated_delta_rule_compute_wy(Tensor q, Tensor k, Tensor v, "
            "Tensor g, Tensor beta, int chunk_size) -> "
            "(Tensor, Tensor, Tensor, Tensor, Tensor)"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def chunk_gated_delta_rule_compute_wy_meta(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            g: torch.Tensor,
            beta: torch.Tensor,
            chunk_size: int,
        ) -> Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]:
            del k, beta, chunk_size
            batch_size, sequence_length, num_key_heads, key_dim = q.shape
            num_value_heads = v.size(2)
            value_dim = v.size(3)
            q_kernel = torch.empty(
                (batch_size, num_key_heads, sequence_length, key_dim),
                dtype=q.dtype,
                device="meta",
            )
            k_kernel = torch.empty_like(q_kernel, device="meta")
            w_kernel = torch.empty(
                (batch_size, num_value_heads, sequence_length, key_dim),
                dtype=q.dtype,
                device="meta",
            )
            u_kernel = torch.empty(
                (batch_size, num_value_heads, sequence_length, value_dim),
                dtype=v.dtype,
                device="meta",
            )
            g_kernel = torch.empty(
                (batch_size, num_value_heads, sequence_length),
                dtype=g.dtype,
                device="meta",
            )
            return q_kernel, k_kernel, w_kernel, u_kernel, g_kernel


_chunk_compute_wy_op_builder = ChunkGatedDeltaRuleComputeWyOpBuilder()
_chunk_compute_wy_op_builder._ensure_initialized()


@impl(get_as_library(), _chunk_compute_wy_op_builder.name, "PrivateUse1")
def _chunk_gated_delta_rule_compute_wy(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    op_module = _chunk_compute_wy_op_builder.load()
    return op_module.chunk_gated_delta_rule_compute_wy(q, k, v, g, beta, chunk_size)


def chunk_gated_delta_rule_compute_wy(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Prepare the chunk-wise WY representation for a Gated Delta Network.

    Args:
        q: Query tensor with shape ``[B, T, Hk, K]`` and dtype ``float16``.
        k: Key tensor with the same shape and dtype as ``q``.
        v: Value tensor with shape ``[B, T, Hv, V]`` and dtype ``float16``.
        g: Log-decay tensor with shape ``[B, T, Hv]`` and dtype ``float32``.
        beta: Update-rate tensor with shape ``[B, T, Hv]`` and dtype ``float16``.
        chunk_size: Chunk length. Currently only ``64`` is supported.

    Returns:
        ``q_kernel``, ``k_kernel``, ``w_kernel``, ``u_kernel`` and ``g_kernel``
        in the layouts required by the following chunk scan.
    """
    return torch.ops.cann_ops_transformer.chunk_gated_delta_rule_compute_wy(
        q, k, v, g, beta, chunk_size
    )
