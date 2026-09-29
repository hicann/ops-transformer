# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


from typing import Optional

import torch
import torch_npu
from torch.library import impl
from cann_ops_transformer.op_builder import OpBuilder, get_as_library

MQSMLA_METADATA_OP_NAME = "ds41.mixed_quant_sparse_flash_mla_metadata"


class MixedQuantSparseFlashMlaOpBuilder(OpBuilder):
    def __init__(self):
        super(MixedQuantSparseFlashMlaOpBuilder, self).__init__(
            "ds41.mixed_quant_sparse_flash_mla", category="attention"
        )

    def sources(self):
        """Python DSL implementation; no C++ extension is compiled."""
        return []

    def schema(self):
        """PyTorch operator signatures."""
        return [
            (
                "ds41.mixed_quant_sparse_flash_mla_metadata(Tensor ori_topk_length, "
                "Tensor cmp_topk_length, "
                "*, "
                "Tensor? cu_seqlens_q=None, "
                "Tensor? seqused_q=None, "
                "Tensor? seqused_ori_kv=None, "
                "Tensor? seqused_cmp_kv=None, "
                "int? batch_size=None, "
                "int? max_seqlen_q=None, "
                "int? max_seqlen_ori_kv=None, "
                "int? max_seqlen_cmp_kv=None, "
                "int num_heads_q, "
                "int num_heads_kv, "
                "int head_dim, "
                "int quant_mode, "
                'str layout_q="TND", '
                'str layout_kv="PA_BBND", '
                "bool has_ori_kv=True, "
                "bool has_cmp_kv=True) -> Tensor"
            ),
            (
                "ds41.mixed_quant_sparse_flash_mla(Tensor q, "
                "*, "
                "Tensor? ori_kv=None, "
                "Tensor? cmp_kv=None, "
                "Tensor? ori_sparse_indices=None, "
                "Tensor? cmp_sparse_indices=None, "
                "Tensor? ori_block_table=None, "
                "Tensor? cmp_block_table=None, "
                "Tensor? cu_seqlens_q=None, "
                "Tensor? seqused_q=None, "
                "Tensor? seqused_ori_kv=None, "
                "Tensor? seqused_cmp_kv=None, "
                "Tensor? ori_topk_length=None, "
                "Tensor? cmp_topk_length=None, "
                "Tensor? sinks=None, "
                "Tensor? metadata=None, "
                "int quant_mode, "
                "float? softmax_scale=None, "
                'str layout_q="TND", '
                'str layout_kv="PA_BBND", '
                "bool return_softmax_lse=False) -> (Tensor, "
                "Tensor)"
            ),
        ]

    def register_meta(self):
        """Register metadata FakeTensor and attention Meta implementations."""

        @torch.library.register_fake("cann_ops_transformer::" + MQSMLA_METADATA_OP_NAME)
        def mixed_quant_sparse_flash_mla_metadata_meta(
            ori_topk_length: torch.Tensor,
            cmp_topk_length: torch.Tensor,
            *,
            cu_seqlens_q: Optional[torch.Tensor] = None,
            seqused_q: Optional[torch.Tensor] = None,
            seqused_ori_kv: Optional[torch.Tensor] = None,
            seqused_cmp_kv: Optional[torch.Tensor] = None,
            batch_size: Optional[int] = None,
            max_seqlen_q: Optional[int] = None,
            max_seqlen_ori_kv: Optional[int] = None,
            max_seqlen_cmp_kv: Optional[int] = None,
            num_heads_q: int,
            num_heads_kv: int,
            head_dim: int,
            quant_mode: int,
            layout_q: str = "TND",
            layout_kv: str = "PA_BBND",
            has_ori_kv: bool = True,
            has_cmp_kv: bool = True,
        ):
            MQSMLA_METADATA_TOTAL_SIZE = 1024
            return ori_topk_length.new_empty(
                (MQSMLA_METADATA_TOTAL_SIZE,), dtype=torch.int32
            )

        @impl(get_as_library(), self.name, "Meta")
        def mixed_quant_sparse_flash_mla_meta(
            q,
            *,
            ori_kv=None,
            cmp_kv=None,
            ori_sparse_indices=None,
            cmp_sparse_indices=None,
            ori_block_table=None,
            cmp_block_table=None,
            cu_seqlens_q=None,
            seqused_q=None,
            seqused_ori_kv=None,
            seqused_cmp_kv=None,
            ori_topk_length=None,
            cmp_topk_length=None,
            sinks=None,
            metadata=None,
            quant_mode,
            softmax_scale=None,
            layout_q="TND",
            layout_kv="PA_BBND",
            return_softmax_lse=False,
        ):
            attn_out = torch.empty(q.shape, dtype=torch.bfloat16, device="meta")
            if return_softmax_lse:
                n2 = ori_kv.shape[2]
                if layout_q == "TND":
                    lse_shape = (n2, q.shape[0], q.shape[1] // n2)
                else:
                    lse_shape = (q.shape[0], n2, q.shape[1], q.shape[2] // n2)
            else:
                lse_shape = (0,)
            softmax_lse = torch.empty(lse_shape, dtype=torch.float32, device="meta")
            return attn_out, softmax_lse


mixed_quant_sparse_flash_mla_op_builder = MixedQuantSparseFlashMlaOpBuilder()
mixed_quant_sparse_flash_mla_op_builder._ensure_initialized()


@impl(get_as_library(), MQSMLA_METADATA_OP_NAME, "PrivateUse1")
def mixed_quant_sparse_flash_mla_metadata(
    ori_topk_length: torch.Tensor,
    cmp_topk_length: torch.Tensor,
    *,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_ori_kv: Optional[torch.Tensor] = None,
    seqused_cmp_kv: Optional[torch.Tensor] = None,
    batch_size: Optional[int] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_ori_kv: Optional[int] = None,
    max_seqlen_cmp_kv: Optional[int] = None,
    num_heads_q: int,
    num_heads_kv: int,
    head_dim: int,
    quant_mode: int,
    layout_q: str = "TND",
    layout_kv: str = "PA_BBND",
    has_ori_kv: bool = True,
    has_cmp_kv: bool = True,
):
    """Generate the core task table through the net wheel DSL host interface."""
    from ops.mixed_quant_sparse_flash_mla import (
        mixed_quant_sparse_flash_mla_metadata as dsl_metadata,
    )

    return dsl_metadata(
        ori_topk_length,
        cmp_topk_length,
        cu_seqlens_q=cu_seqlens_q,
        seqused_q=seqused_q,
        seqused_ori_kv=seqused_ori_kv,
        seqused_cmp_kv=seqused_cmp_kv,
        batch_size=batch_size,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_ori_kv=max_seqlen_ori_kv,
        max_seqlen_cmp_kv=max_seqlen_cmp_kv,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        head_dim=head_dim,
        quant_mode=quant_mode,
        layout_q=layout_q,
        layout_kv=layout_kv,
        has_ori_kv=has_ori_kv,
        has_cmp_kv=has_cmp_kv,
    )


@torch.library.register_kernel("cann_ops_transformer::" + MQSMLA_METADATA_OP_NAME, None)
def mixed_quant_sparse_flash_mla_metadata_fallback(
    ori_topk_length: torch.Tensor,
    cmp_topk_length: torch.Tensor,
    *,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_ori_kv: Optional[torch.Tensor] = None,
    seqused_cmp_kv: Optional[torch.Tensor] = None,
    batch_size: Optional[int] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_ori_kv: Optional[int] = None,
    max_seqlen_cmp_kv: Optional[int] = None,
    num_heads_q: int,
    num_heads_kv: int,
    head_dim: int,
    quant_mode: int,
    layout_q: str = "TND",
    layout_kv: str = "PA_BBND",
    has_ori_kv: bool = True,
    has_cmp_kv: bool = True,
):
    return mixed_quant_sparse_flash_mla_metadata(
        ori_topk_length,
        cmp_topk_length,
        cu_seqlens_q=cu_seqlens_q,
        seqused_q=seqused_q,
        seqused_ori_kv=seqused_ori_kv,
        seqused_cmp_kv=seqused_cmp_kv,
        batch_size=batch_size,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_ori_kv=max_seqlen_ori_kv,
        max_seqlen_cmp_kv=max_seqlen_cmp_kv,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        head_dim=head_dim,
        quant_mode=quant_mode,
        layout_q=layout_q,
        layout_kv=layout_kv,
        has_ori_kv=has_ori_kv,
        has_cmp_kv=has_cmp_kv,
    )


torch.compiler.allow_in_graph(mixed_quant_sparse_flash_mla_metadata)


@impl(get_as_library(), mixed_quant_sparse_flash_mla_op_builder.name, "PrivateUse1")
def mixed_quant_sparse_flash_mla(
    q,
    *,
    ori_kv=None,
    cmp_kv=None,
    ori_sparse_indices=None,
    cmp_sparse_indices=None,
    ori_block_table=None,
    cmp_block_table=None,
    cu_seqlens_q=None,
    seqused_q=None,
    seqused_ori_kv=None,
    seqused_cmp_kv=None,
    ori_topk_length=None,
    cmp_topk_length=None,
    sinks=None,
    metadata=None,
    quant_mode,
    softmax_scale=None,
    layout_q="TND",
    layout_kv="PA_BBND",
    return_softmax_lse=False,
):
    """Allocate outputs and pass the original inputs to the net wheel DSL."""
    from ops.mixed_quant_sparse_flash_mla import (
        mixed_quant_sparse_flash_mla as dsl_attention,
    )

    attn_out = torch.empty(q.shape, dtype=torch.bfloat16, device=q.device)
    if return_softmax_lse:
        n2 = ori_kv.shape[2]
        if layout_q == "TND":
            lse_shape = (n2, q.shape[0], q.shape[1] // n2)
        else:
            lse_shape = (q.shape[0], n2, q.shape[1], q.shape[2] // n2)
    else:
        lse_shape = (0,)
    softmax_lse = torch.empty(lse_shape, dtype=torch.float32, device=q.device)
    dsl_attention(
        q,
        ori_kv=ori_kv,
        cmp_kv=cmp_kv,
        ori_sparse_indices=ori_sparse_indices,
        cmp_sparse_indices=cmp_sparse_indices,
        ori_block_table=ori_block_table,
        cmp_block_table=cmp_block_table,
        cu_seqlens_q=cu_seqlens_q,
        seqused_q=seqused_q,
        seqused_ori_kv=seqused_ori_kv,
        seqused_cmp_kv=seqused_cmp_kv,
        ori_topk_length=ori_topk_length,
        cmp_topk_length=cmp_topk_length,
        sinks=sinks,
        metadata=metadata,
        quant_mode=quant_mode,
        softmax_scale=softmax_scale,
        layout_q=layout_q,
        layout_kv=layout_kv,
        return_softmax_lse=return_softmax_lse,
        out=attn_out,
        lse=softmax_lse,
    )
    return attn_out, softmax_lse
