# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


import torch
import torch_npu
from torch.library import impl
from cann_ops_transformer.op_builder import OpBuilder, get_as_library


class QuantLightningIndexerOpBuilder(OpBuilder):
    def __init__(self):
        super(QuantLightningIndexerOpBuilder, self).__init__(
            "ds41.quant_lightning_indexer", category="attention"
        )

    def sources(self):
        """Python DSL implementation; no C++ extension is compiled."""
        return []

    def schema(self):
        """PyTorch signatures aligned with the interface workbooks."""
        return [
            (
                "ds41.quant_lightning_indexer_metadata(Tensor? cu_seqlens_q=None, Tensor? "
                "cu_seqlens_k=None, Tensor? seqused_q=None, Tensor? seqused_k=None, Tensor? "
                "cmp_residual_k=None, *, int? batch_size=None, int max_seqlen_q=-1, int max_seqlen_k=-1, "
                "int num_heads_q, int num_heads_k, int head_dim, int topk, int mask_mode=0, int "
                'cmp_ratio=1, str layout_q="TND", str layout_k="TND", int candidate_topk_blocks=-1, int '
                "candidate_block_size=-1) -> Tensor"
            ),
            (
                "ds41.quant_lightning_indexer(Tensor q, Tensor k, Tensor w, Tensor descale_q, Tensor "
                "descale_k, int topk, int quant_mode, *, Tensor? cu_seqlens_q=None, Tensor? "
                "cu_seqlens_k=None, Tensor? seqused_q=None, Tensor? seqused_k=None, Tensor? "
                "cmp_residual_k=None, Tensor? block_table=None, Tensor? output_idx_offset=None, Tensor? "
                "metadata=None, int max_seqlen_q=-1, int mask_mode=0, int cmp_ratio=1, str "
                'layout_q="TND", str layout_k="TND", bool return_value=False, int '
                "candidate_topk_blocks=-1, int candidate_block_size=-1) -> (Tensor, Tensor, Tensor, "
                "Tensor)"
            ),
        ]

    def register_meta(self):
        """Register the FakeTensor/Meta output implementation."""

        @torch.library.register_fake(
            "cann_ops_transformer::ds41.quant_lightning_indexer_metadata"
        )
        def quant_lightning_indexer_metadata_meta(
            cu_seqlens_q=None,
            cu_seqlens_k=None,
            seqused_q=None,
            seqused_k=None,
            cmp_residual_k=None,
            *,
            batch_size=None,
            max_seqlen_q=-1,
            max_seqlen_k=-1,
            num_heads_q,
            num_heads_k,
            head_dim,
            topk,
            mask_mode=0,
            cmp_ratio=1,
            layout_q="TND",
            layout_k="TND",
            candidate_topk_blocks=-1,
            candidate_block_size=-1,
        ):
            if layout_q != "TND" or layout_k not in ("PA_BBND", "TND"):
                raise ValueError(
                    "layout_q must be TND and layout_k must be PA_BBND or TND"
                )
            device = next(
                (
                    t.device
                    for t in (
                        cu_seqlens_q,
                        cu_seqlens_k,
                        seqused_q,
                        seqused_k,
                        cmp_residual_k,
                    )
                    if t is not None
                ),
                None,
            )
            if device is None:
                device = torch.device("npu", torch.npu.current_device())
            return torch.empty((1024,), dtype=torch.int32, device=device)

        @impl(get_as_library(), self.name, "Meta")
        def quant_lightning_indexer_meta(
            q,
            k,
            w,
            descale_q,
            descale_k,
            topk,
            quant_mode,
            *,
            cu_seqlens_q=None,
            cu_seqlens_k=None,
            seqused_q=None,
            seqused_k=None,
            cmp_residual_k=None,
            block_table=None,
            output_idx_offset=None,
            metadata=None,
            max_seqlen_q=-1,
            mask_mode=0,
            cmp_ratio=1,
            layout_q="TND",
            layout_k="TND",
            return_value=False,
            candidate_topk_blocks=-1,
            candidate_block_size=-1,
        ):
            if layout_q != "TND" or layout_k not in ("PA_BBND", "TND"):
                raise ValueError(
                    "layout_q must be TND and layout_k must be PA_BBND or TND"
                )
            heads = k.shape[1] if layout_k == "TND" else k.shape[2]
            output_shape = (q.shape[0], heads, topk)
            sparse_indices = torch.empty(output_shape, dtype=torch.int32, device="meta")
            sparse_values_shape = output_shape if return_value else (0,)
            sparse_values = torch.empty(
                sparse_values_shape, dtype=torch.bfloat16, device="meta"
            )
            if candidate_topk_blocks > 0:
                candidate_indices_shape = (
                    q.shape[0],
                    heads,
                    candidate_topk_blocks,
                )
                candidate_length_shape = (q.shape[0], heads)
            else:
                candidate_indices_shape = (0,)
                candidate_length_shape = (0,)
            candidate_block_indices = torch.empty(
                candidate_indices_shape, dtype=torch.int32, device="meta"
            )
            candidate_block_length = torch.empty(
                candidate_length_shape, dtype=torch.int32, device="meta"
            )
            return (
                sparse_indices,
                sparse_values,
                candidate_block_indices,
                candidate_block_length,
            )


quant_lightning_indexer_op_builder = QuantLightningIndexerOpBuilder()
quant_lightning_indexer_op_builder._ensure_initialized()


@impl(get_as_library(), quant_lightning_indexer_op_builder.name, "PrivateUse1")
def quant_lightning_indexer(
    q,
    k,
    w,
    descale_q,
    descale_k,
    topk,
    quant_mode,
    *,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_q=None,
    seqused_k=None,
    cmp_residual_k=None,
    block_table=None,
    output_idx_offset=None,
    metadata=None,
    max_seqlen_q=-1,
    mask_mode=0,
    cmp_ratio=1,
    layout_q="TND",
    layout_k="TND",
    return_value=False,
    candidate_topk_blocks=-1,
    candidate_block_size=-1,
):
    """Run the CANNBotDSL QLI kernel through the Torch NPU dispatcher."""
    from ops.quant_lightning_indexer_dsl import (
        quant_lightning_indexer as dsl_quant_lightning_indexer,
    )

    return dsl_quant_lightning_indexer(
        q,
        k,
        w,
        descale_q,
        descale_k,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        cmp_residual_k=cmp_residual_k,
        block_table=block_table,
        output_idx_offset=output_idx_offset,
        metadata=metadata,
        topk=topk,
        quant_mode=quant_mode,
        max_seqlen_q=max_seqlen_q,
        mask_mode=mask_mode,
        cmp_ratio=cmp_ratio,
        layout_q=layout_q,
        layout_k=layout_k,
        return_value=return_value,
        candidate_topk_blocks=candidate_topk_blocks,
        candidate_block_size=candidate_block_size,
    )


@impl(get_as_library(), "ds41.quant_lightning_indexer_metadata", "PrivateUse1")
def _quant_lightning_indexer_metadata_impl(
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_q=None,
    seqused_k=None,
    cmp_residual_k=None,
    *,
    batch_size=None,
    max_seqlen_q=-1,
    max_seqlen_k=-1,
    num_heads_q,
    num_heads_k,
    head_dim,
    topk,
    mask_mode=0,
    cmp_ratio=1,
    layout_q="TND",
    layout_k="TND",
    candidate_topk_blocks=-1,
    candidate_block_size=-1,
):
    from ops.quant_lightning_indexer_metadata_dsl import (
        quant_lightning_indexer_metadata as dsl_metadata,
    )

    return dsl_metadata(
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        cmp_residual_k,
        batch_size=batch_size,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        num_heads_q=num_heads_q,
        num_heads_k=num_heads_k,
        head_dim=head_dim,
        topk=topk,
        mask_mode=mask_mode,
        cmp_ratio=cmp_ratio,
        layout_q=layout_q,
        layout_k=layout_k,
        candidate_topk_blocks=candidate_topk_blocks,
        candidate_block_size=candidate_block_size,
    )


@torch.library.register_kernel(
    "cann_ops_transformer::ds41.quant_lightning_indexer_metadata", None
)
def quant_lightning_indexer_metadata_fallback(
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_q=None,
    seqused_k=None,
    cmp_residual_k=None,
    *,
    batch_size=None,
    max_seqlen_q=-1,
    max_seqlen_k=-1,
    num_heads_q,
    num_heads_k,
    head_dim,
    topk,
    mask_mode=0,
    cmp_ratio=1,
    layout_q="TND",
    layout_k="TND",
    candidate_topk_blocks=-1,
    candidate_block_size=-1,
):
    """Route calls without a backend Tensor to the NPU implementation."""
    return _quant_lightning_indexer_metadata_impl(
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        cmp_residual_k,
        batch_size=batch_size,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        num_heads_q=num_heads_q,
        num_heads_k=num_heads_k,
        head_dim=head_dim,
        topk=topk,
        mask_mode=mask_mode,
        cmp_ratio=cmp_ratio,
        layout_q=layout_q,
        layout_k=layout_k,
        candidate_topk_blocks=candidate_topk_blocks,
        candidate_block_size=candidate_block_size,
    )


def quant_lightning_indexer_metadata(
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_q=None,
    seqused_k=None,
    cmp_residual_k=None,
    *,
    batch_size=None,
    max_seqlen_q=-1,
    max_seqlen_k=-1,
    num_heads_q,
    num_heads_k,
    head_dim,
    topk,
    mask_mode=0,
    cmp_ratio=1,
    layout_q="TND",
    layout_k="TND",
    candidate_topk_blocks=-1,
    candidate_block_size=-1,
):
    """Dispatch through the registered operator for eager and graph execution."""
    return torch.ops.cann_ops_transformer.ds41.quant_lightning_indexer_metadata(
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        cmp_residual_k,
        batch_size=batch_size,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        num_heads_q=num_heads_q,
        num_heads_k=num_heads_k,
        head_dim=head_dim,
        topk=topk,
        mask_mode=mask_mode,
        cmp_ratio=cmp_ratio,
        layout_q=layout_q,
        layout_k=layout_k,
        candidate_topk_blocks=candidate_topk_blocks,
        candidate_block_size=candidate_block_size,
    )
