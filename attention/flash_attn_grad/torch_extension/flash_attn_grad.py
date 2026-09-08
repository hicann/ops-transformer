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


class FlashAttnGradOpBuilder(OpBuilder):
    def __init__(self):
        super(FlashAttnGradOpBuilder, self).__init__(
            "flash_attn_grad", category="attention"
        )

    def sources(self):
        """Path to C++ source code."""
        return ["csrc/attention/flash_attn_grad.cpp"]

    def schema(self) -> str:
        """PyTorch operator signature."""
        return [
            "flash_attn_grad(Tensor q, Tensor k, Tensor v, Tensor dout, Tensor attn_out, Tensor softmax_lse, "
            "Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_kv=None, "
            "Tensor? seqused_q=None, Tensor? seqused_kv=None, "
            "Tensor? sinks=None, Tensor? attn_mask=None, Tensor? metadata=None, "
            "float softmax_scale=0.0, int mask_mode=0, int win_left=-1, int win_right=-1, "
            "int max_seqlen_q=-1, int max_seqlen_kv=-1, "
            'str layout_q="BSND", str layout_kv="BSND", str layout_out="BSND") '
            "-> (Tensor, Tensor, Tensor)",
        ]

    def register_meta(self):
        """
        Registers the Meta implementation (Shape/Dtype inference).
        dq shape = q shape, dk shape = k shape, dv shape = v shape.
        """

        @impl(get_as_library(), self.name, "Meta")
        def flash_attn_grad_meta(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            dout: torch.Tensor,
            attn_out: torch.Tensor,
            softmax_lse: torch.Tensor,
            cu_seqlens_q: Optional[torch.Tensor] = None,
            cu_seqlens_kv: Optional[torch.Tensor] = None,
            seqused_q: Optional[torch.Tensor] = None,
            seqused_kv: Optional[torch.Tensor] = None,
            sinks: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
            metadata: Optional[torch.Tensor] = None,
            softmax_scale: Optional[float] = 0.0,
            mask_mode: Optional[int] = 0,
            win_left: Optional[int] = -1,
            win_right: Optional[int] = -1,
            max_seqlen_q: Optional[int] = -1,
            max_seqlen_kv: Optional[int] = -1,
            layout_q: Optional[str] = "BSND",
            layout_kv: Optional[str] = "BSND",
            layout_out: Optional[str] = "BSND",
        ):
            dq = torch.empty(q.size(), dtype=q.dtype, device="meta")
            dk = torch.empty(k.size(), dtype=k.dtype, device="meta")
            dv = torch.empty(v.size(), dtype=v.dtype, device="meta")
            return (dq, dk, dv)


flash_attn_grad_op_builder = FlashAttnGradOpBuilder()
flash_attn_grad_op_builder._ensure_initialized()


@impl(get_as_library(), flash_attn_grad_op_builder.name, "PrivateUse1")
def flash_attn_grad(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dout: torch.Tensor,
    attn_out: torch.Tensor,
    softmax_lse: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_kv: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_kv: Optional[torch.Tensor] = None,
    sinks: Optional[torch.Tensor] = None,
    attn_mask: Optional[torch.Tensor] = None,
    metadata: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = 0.0,
    mask_mode: Optional[int] = 0,
    win_left: Optional[int] = -1,
    win_right: Optional[int] = -1,
    max_seqlen_q: Optional[int] = -1,
    max_seqlen_kv: Optional[int] = -1,
    layout_q: Optional[str] = "BSND",
    layout_kv: Optional[str] = "BSND",
    layout_out: Optional[str] = "BSND",
):
    """
    Flash Attention backward gradient computation.

    Computes gradients dq, dk, dv for the flash attention forward pass.

    Args:
        q: Query tensor of shape (B, S1, N1, D) for BSND or (T1, N1, D) for TND.
        k: Key tensor of shape (B, S2, N2, D) for BSND or (T2, N2, D) for TND.
        v: Value tensor, same shape as k.
        dout: Upstream gradient, same shape as q.
        attn_out: Attention output from forward pass, same shape as v.
        softmax_lse: Softmax LSE from forward pass.
        cu_seqlens_q: Cumulative sequence lengths for Q (TND layout).
        cu_seqlens_kv: Cumulative sequence lengths for KV (TND layout).
        seqused_q: Actual sequence lengths used for Q.
        seqused_kv: Actual sequence lengths used for KV.
        sinks: Sink tensor.
        attn_mask: Attention mask tensor (for mask_mode 3 or 4).
        metadata: Precomputed metadata tensor from flash_attn_metadata.
        softmax_scale: Softmax scale factor. 0.0 means 1/sqrt(head_dim).
        mask_mode: 0 for full compute, 3 for causal, 4 for windoutw.
        win_left: Left windoutw size for mask_mode 4. -1 means infinite.
        win_right: Right windoutw size for mask_mode 4. -1 means infinite.
        max_seqlen_q: Maximum sequence length for Q. -1 means unknown.
        max_seqlen_kv: Maximum sequence length for KV. -1 means unknown.
        layout_q: Layout of Q tensor ("BSND", "TND", or "BNSD").
        layout_kv: Layout of KV tensor ("BSND", "TND", or "BNSD").
        layout_out: Layout of output tensors ("BSND", "TND", or "BNSD").

    Returns:
        A tuple of (dq, dk, dv) gradient tensors.
    """
    op_module = flash_attn_grad_op_builder.load()
    return op_module.flash_attn_grad(
        q,
        k,
        v,
        dout,
        attn_out,
        softmax_lse,
        cu_seqlens_q,
        cu_seqlens_kv,
        seqused_q,
        seqused_kv,
        sinks,
        attn_mask,
        metadata,
        softmax_scale,
        mask_mode,
        win_left,
        win_right,
        max_seqlen_q,
        max_seqlen_kv,
        layout_q,
        layout_kv,
        layout_out,
    )


flash_attn_grad = torch.ops.cann_ops_transformer.flash_attn_grad
