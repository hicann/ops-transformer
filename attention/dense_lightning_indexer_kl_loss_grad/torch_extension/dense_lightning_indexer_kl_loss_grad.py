# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import Optional

import torch
from torch.library import impl

from cann_ops_transformer.op_builder import OpBuilder, get_as_library


DLI_KL_LOSS_GRAD_METADATA_SIZE = 64
DLI_KL_LOSS_GRAD_METADATA_MAX_CORE_NUM = 25
DLI_KL_LOSS_GRAD_METADATA_OP_NAME = "dense_lightning_indexer_kl_loss_grad_metadata"


class DenseLightningIndexerKLLossGradOpBuilder(OpBuilder):
    def __init__(self):
        super(DenseLightningIndexerKLLossGradOpBuilder, self).__init__(
            "dense_lightning_indexer_kl_loss_grad", category="attention"
        )

    def sources(self):
        return ["csrc/attention/dense_lightning_indexer_kl_loss_grad.cpp"]

    def schema(self) -> str:
        return [
            "dense_lightning_indexer_kl_loss_grad_metadata("
            "int num_heads_q, int num_heads_k, int head_dim, *, "
            "Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_k=None, Tensor? seqused_q=None, "
            "Tensor? seqused_k=None, Tensor? cmp_residual_k=None, int? batch_size=None, "
            "int? max_seqlen_q=None, int? max_seqlen_k=None, "
            "str? layout_q=None, str? layout_k=None, int? mask_mode=None, "
            "int? cmp_ratio=None) -> Tensor",
            "dense_lightning_indexer_kl_loss_grad("
            "Tensor q, Tensor k, Tensor w, Tensor attn_softmax_l1_norm, Tensor softmax_lse, "
            "*, Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_k=None, Tensor? seqused_q=None, "
            "Tensor? seqused_k=None, Tensor? cmp_residual_k=None, Tensor? metadata=None, "
            'str layout_q="TND", str layout_k="TND", int mask_mode=3, '
            "int cmp_ratio=1) -> (Tensor, Tensor, Tensor, Tensor)",
        ]

    def register_meta(self):
        @torch.library.register_fake(
            "cann_ops_transformer::" + DLI_KL_LOSS_GRAD_METADATA_OP_NAME
        )
        def dense_lightning_indexer_kl_loss_grad_metadata_meta(
            num_heads_q,
            num_heads_k,
            head_dim,
            *,
            cu_seqlens_q=None,
            cu_seqlens_k=None,
            seqused_q=None,
            seqused_k=None,
            cmp_residual_k=None,
            batch_size=None,
            max_seqlen_q=None,
            max_seqlen_k=None,
            layout_q=None,
            layout_k=None,
            mask_mode=None,
            cmp_ratio=None,
        ):
            return torch.empty(
                (DLI_KL_LOSS_GRAD_METADATA_SIZE,), dtype=torch.int32, device="meta"
            )

        @impl(get_as_library(), self.name, "Meta")
        def dense_lightning_indexer_kl_loss_grad_meta(
            q,
            k,
            w,
            attn_softmax_l1_norm,
            softmax_lse,
            *,
            cu_seqlens_q=None,
            cu_seqlens_k=None,
            seqused_q=None,
            seqused_k=None,
            cmp_residual_k=None,
            metadata=None,
            layout_q="BSND",
            layout_k="BSND",
            mask_mode=0,
            cmp_ratio=1,
        ):
            dq = q.new_empty(q.shape, dtype=q.dtype, device="meta")
            dk = k.new_empty(k.shape, dtype=k.dtype, device="meta")
            dw = w.new_empty(w.shape, dtype=w.dtype, device="meta")
            softmax_out = attn_softmax_l1_norm.new_empty(
                attn_softmax_l1_norm.shape,
                dtype=attn_softmax_l1_norm.dtype,
                device="meta",
            )
            return dq, dk, dw, softmax_out


dense_lightning_indexer_kl_loss_grad_op_builder = (
    DenseLightningIndexerKLLossGradOpBuilder()
)
dense_lightning_indexer_kl_loss_grad_op_builder._ensure_initialized()


@impl(get_as_library(), DLI_KL_LOSS_GRAD_METADATA_OP_NAME, "PrivateUse1")
def dense_lightning_indexer_kl_loss_grad_metadata(
    num_heads_q,
    num_heads_k,
    head_dim,
    *,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_q=None,
    seqused_k=None,
    cmp_residual_k=None,
    batch_size=None,
    max_seqlen_q=None,
    max_seqlen_k=None,
    layout_q=None,
    layout_k=None,
    mask_mode=None,
    cmp_ratio=None,
):
    batch_size = 0 if batch_size is None else batch_size
    max_seqlen_q = 0 if max_seqlen_q is None else max_seqlen_q
    max_seqlen_k = 0 if max_seqlen_k is None else max_seqlen_k
    layout_q = "BSND" if layout_q is None else layout_q
    layout_k = "BSND" if layout_k is None else layout_k
    mask_mode = 0 if mask_mode is None else mask_mode
    cmp_ratio = 1 if cmp_ratio is None else cmp_ratio

    op_module = dense_lightning_indexer_kl_loss_grad_op_builder.load()
    return op_module.dense_lightning_indexer_kl_loss_grad_metadata(
        num_heads_q,
        num_heads_k,
        head_dim,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        cmp_residual_k,
        batch_size,
        max_seqlen_q,
        max_seqlen_k,
        layout_q,
        layout_k,
        mask_mode,
        cmp_ratio,
    )


@torch.library.register_kernel(
    "cann_ops_transformer::" + DLI_KL_LOSS_GRAD_METADATA_OP_NAME, None
)
def dense_lightning_indexer_kl_loss_grad_metadata_fallback(
    num_heads_q,
    num_heads_k,
    head_dim,
    *,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_q=None,
    seqused_k=None,
    cmp_residual_k=None,
    batch_size=None,
    max_seqlen_q=None,
    max_seqlen_k=None,
    layout_q=None,
    layout_k=None,
    mask_mode=None,
    cmp_ratio=None,
):
    return dense_lightning_indexer_kl_loss_grad_metadata(
        num_heads_q,
        num_heads_k,
        head_dim,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        cmp_residual_k=cmp_residual_k,
        batch_size=batch_size,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        layout_q=layout_q,
        layout_k=layout_k,
        mask_mode=mask_mode,
        cmp_ratio=cmp_ratio,
    )


torch.compiler.allow_in_graph(dense_lightning_indexer_kl_loss_grad_metadata)


@impl(
    get_as_library(),
    dense_lightning_indexer_kl_loss_grad_op_builder.name,
    "PrivateUse1",
)
def dense_lightning_indexer_kl_loss_grad(
    q,
    k,
    w,
    attn_softmax_l1_norm,
    softmax_lse,
    *,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_q=None,
    seqused_k=None,
    cmp_residual_k=None,
    metadata=None,
    layout_q="BSND",
    layout_k="BSND",
    mask_mode=0,
    cmp_ratio=1,
):
    op_module = dense_lightning_indexer_kl_loss_grad_op_builder.load()
    return op_module.dense_lightning_indexer_kl_loss_grad(
        q,
        k,
        w,
        attn_softmax_l1_norm,
        softmax_lse,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        cmp_residual_k,
        metadata,
        layout_q,
        layout_k,
        mask_mode,
        cmp_ratio,
    )


try:
    import torchair
    from torchair._ge_concrete_graph.fx2ge_converter import (
        register_fx_node_ge_converter,
    )
    from torchair.ge._ge_graph import Tensor, TensorSpec
    from torchair.ge import attr

    _TORCHAIR_AVAILABLE = True
except ImportError:
    _TORCHAIR_AVAILABLE = False


if _TORCHAIR_AVAILABLE:

    @register_fx_node_ge_converter(
        torch.ops.cann_ops_transformer.dense_lightning_indexer_kl_loss_grad.default
    )
    def convert_dense_lightning_indexer_kl_loss_grad(
        q: Tensor,
        k: Tensor,
        w: Tensor,
        attn_softmax_l1_norm: Tensor,
        softmax_lse: Tensor,
        *,
        cu_seqlens_q: Optional[Tensor] = None,
        cu_seqlens_k: Optional[Tensor] = None,
        seqused_q: Optional[Tensor] = None,
        seqused_k: Optional[Tensor] = None,
        cmp_residual_k: Optional[Tensor] = None,
        metadata: Optional[Tensor] = None,
        layout_q: str = "BSND",
        layout_k: str = "BSND",
        mask_mode: int = 0,
        cmp_ratio: int = 1,
        meta_outputs: TensorSpec = None,
    ):
        return torchair.ge.custom_op(
            "DenseLightningIndexerKLLossGrad",
            inputs={
                "q": q,
                "k": k,
                "w": w,
                "attn_softmax_l1_norm": attn_softmax_l1_norm,
                "softmax_lse": softmax_lse,
                "cu_seqlens_q": cu_seqlens_q,
                "cu_seqlens_k": cu_seqlens_k,
                "seqused_q": seqused_q,
                "seqused_k": seqused_k,
                "cmp_residual_k": cmp_residual_k,
                "metadata": metadata,
            },
            attrs={
                "layout_q": attr.Str(layout_q),
                "layout_k": attr.Str(layout_k),
                "mask_mode": attr.Int(mask_mode),
                "cmp_ratio": attr.Int(cmp_ratio),
            },
            outputs=["dq", "dk", "dw", "softmax_out"],
        )

    @register_fx_node_ge_converter(
        torch.ops.cann_ops_transformer.dense_lightning_indexer_kl_loss_grad_metadata.default
    )
    def convert_dense_lightning_indexer_kl_loss_grad_metadata(
        num_heads_q: int,
        num_heads_k: int,
        head_dim: int,
        *,
        cu_seqlens_q: Optional[Tensor] = None,
        cu_seqlens_k: Optional[Tensor] = None,
        seqused_q: Optional[Tensor] = None,
        seqused_k: Optional[Tensor] = None,
        cmp_residual_k: Optional[Tensor] = None,
        batch_size: Optional[int] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_k: Optional[int] = None,
        layout_q: Optional[str] = None,
        layout_k: Optional[str] = None,
        mask_mode: Optional[int] = None,
        cmp_ratio: Optional[int] = None,
        meta_outputs: TensorSpec = None,
    ):
        batch_size = 0 if batch_size is None else batch_size
        max_seqlen_q = 0 if max_seqlen_q is None else max_seqlen_q
        max_seqlen_k = 0 if max_seqlen_k is None else max_seqlen_k
        layout_q = "BSND" if layout_q is None else layout_q
        layout_k = "BSND" if layout_k is None else layout_k
        mask_mode = 0 if mask_mode is None else mask_mode
        cmp_ratio = 1 if cmp_ratio is None else cmp_ratio
        stream_info = torch.npu.get_stream_limit(torch.npu.current_stream())
        aic_core_num = (
            stream_info.get("cube_core_num") or DLI_KL_LOSS_GRAD_METADATA_MAX_CORE_NUM
        )
        return torchair.ge.custom_op(
            "DenseLightningIndexerKLLossGradMetadata",
            inputs={
                "cu_seqlens_q": cu_seqlens_q,
                "cu_seqlens_k": cu_seqlens_k,
                "seqused_q": seqused_q,
                "seqused_k": seqused_k,
                "cmp_residual_k": cmp_residual_k,
            },
            attrs={
                "batch_size": attr.Int(batch_size),
                "max_seqlen_q": attr.Int(max_seqlen_q),
                "max_seqlen_k": attr.Int(max_seqlen_k),
                "num_heads_q": attr.Int(num_heads_q),
                "num_heads_k": attr.Int(num_heads_k),
                "head_dim": attr.Int(head_dim),
                "layout_q": attr.Str(layout_q),
                "layout_k": attr.Str(layout_k),
                "mask_mode": attr.Int(mask_mode),
                "cmp_ratio": attr.Int(cmp_ratio),
                "aic_core_num": attr.Int(aic_core_num),
            },
            outputs=["metadata"],
        )
