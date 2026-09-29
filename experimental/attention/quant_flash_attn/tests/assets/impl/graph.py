#!/usr/bin/python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import logging

import torch

logger = logging.getLogger(__name__)


def _resolve_metadata_op():
    try:
        from cann_ops_transformer.ops import quant_flash_attn_metadata

        return quant_flash_attn_metadata
    except ImportError:
        import torch_npu  # noqa: F401  (ensure op registration)

        return torch.ops.cann_ops_transformer.quant_flash_attn_metadata


def _resolve_main_op():
    try:
        from cann_ops_transformer.ops import quant_flash_attn

        return quant_flash_attn
    except ImportError:
        import torch_npu  # noqa: F401  (ensure op registration)

        return torch.ops.cann_ops_transformer.quant_flash_attn


def _none_if_empty(t):
    if t is None:
        return None
    return t if t.numel() > 0 else None


def _head_index(shape, layout, kv):
    if kv and layout == "PA_BBND":
        return 2
    if not kv and layout == "BSND":
        return 2
    return 1


class QuantFlashAttnMxfp4AclGraph(torch.nn.Module):
    """aclgraph 编译目标: forward 只调 quant_flash_attn 主算子."""

    def __init__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_descale: torch.Tensor,
        k_descale: torch.Tensor,
        v_descale: torch.Tensor,
        quant_mode: int,
        block_table: torch.Tensor = None,
        p_scale: torch.Tensor = None,
        cu_seqlens_q: torch.Tensor = None,
        cu_seqlens_kv: torch.Tensor = None,
        seqused_q: torch.Tensor = None,
        seqused_kv: torch.Tensor = None,
        sinks: torch.Tensor = None,
        attn_mask: torch.Tensor = None,
        metadata: torch.Tensor = None,
        softmax_scale: float = 1.0,
        mask_mode: int = 0,
        win_left: int = -1,
        win_right: int = -1,
        max_seqlen_q: int = -1,
        max_seqlen_kv: int = -1,
        layout_q: str = "BSND",
        layout_q_descale: str = "BSND",
        layout_kv: str = "BSND",
        layout_out: str = "BSND",
        return_softmax_lse: bool = False,
        **kwargs,
    ):
        super().__init__()
        import torch_npu

        torch_npu.npu.set_device(int(kwargs.get("device_id", 0)))

        layout_q = str(layout_q)
        layout_kv = str(layout_kv)

        q_shape = tuple(int(value) for value in q.shape)
        k_shape = tuple(int(value) for value in k.shape)
        head_dim = int(kwargs.get("D") or q_shape[-1])
        num_heads_q = int(
            kwargs.get("N_q") or q_shape[_head_index(q_shape, layout_q, False)]
        )
        num_heads_kv = int(
            kwargs.get("N_kv") or k_shape[_head_index(k_shape, layout_kv, True)]
        )
        head_dim_v = int(kwargs.get("V_D") or head_dim)
        batch_size = kwargs.get("B")

        cu_seqlens_q_t = _none_if_empty(cu_seqlens_q)
        cu_seqlens_kv_t = _none_if_empty(cu_seqlens_kv)
        seqused_q_t = _none_if_empty(seqused_q)
        seqused_kv_t = _none_if_empty(seqused_kv)

        torch.npu.synchronize()

        logger.info("[GRAPH] 构建 metadata (quant_flash_attn_metadata)")
        is_varlen_q = layout_q in ("TND", "NTD")
        if is_varlen_q:
            batch_size_arg = None
        else:
            batch_size_arg = (
                int(batch_size) if batch_size is not None else int(q_shape[0])
            )
        self.metadata = _resolve_metadata_op()(
            num_heads_q=num_heads_q,
            num_heads_kv=num_heads_kv,
            head_dim=head_dim,
            quant_mode=int(quant_mode) if quant_mode is not None else 5,
            cu_seqlens_q=cu_seqlens_q_t,
            cu_seqlens_kv=cu_seqlens_kv_t,
            seqused_q=seqused_q_t,
            seqused_kv=seqused_kv_t,
            batch_size=batch_size_arg,
            max_seqlen_q=int(max_seqlen_q),
            max_seqlen_kv=int(max_seqlen_kv),
            head_dim_v=head_dim_v,
            mask_mode=int(mask_mode),
            win_left=int(win_left),
            win_right=int(win_right),
            layout_q=layout_q,
            layout_q_descale=str(layout_q_descale),
            layout_kv=layout_kv,
            layout_out=str(layout_out),
            is_grad_enabled=False,
        )
        if self.metadata.device != q.device:
            self.metadata = self.metadata.to(q.device)

        self.q = q
        self.k = k
        self.v = v
        self.q_descale = q_descale
        self.k_descale = k_descale
        self.v_descale = v_descale
        self.quant_mode = int(quant_mode) if quant_mode is not None else 5
        self.block_table = block_table
        self.p_scale = p_scale
        self.cu_seqlens_q = cu_seqlens_q_t
        self.cu_seqlens_kv = cu_seqlens_kv_t
        self.seqused_q = seqused_q_t
        self.seqused_kv = seqused_kv_t
        self.sinks = sinks
        self.attn_mask = attn_mask
        self.softmax_scale = softmax_scale
        self.mask_mode = int(mask_mode)
        self.win_left = int(win_left)
        self.win_right = int(win_right)
        self.max_seqlen_q = int(max_seqlen_q)
        self.max_seqlen_kv = int(max_seqlen_kv)
        self.layout_q = layout_q
        self.layout_q_descale = str(layout_q_descale)
        self.layout_kv = layout_kv
        self.layout_out = str(layout_out)
        self.return_softmax_lse = bool(return_softmax_lse)

        logger.info(
            "[GRAPH] __init__ done: q=%s, k=%s, v=%s, metadata=%s",
            self.q.shape,
            self.k.shape,
            self.v.shape,
            self.metadata.shape,
        )

    def forward(self):
        atten_out, lse_out = _resolve_main_op()(
            self.q,
            self.k,
            self.v,
            self.q_descale,
            self.k_descale,
            self.v_descale,
            self.quant_mode,
            block_table=self.block_table,
            p_scale=self.p_scale,
            cu_seqlens_q=self.cu_seqlens_q,
            cu_seqlens_kv=self.cu_seqlens_kv,
            seqused_q=self.seqused_q,
            seqused_kv=self.seqused_kv,
            sinks=self.sinks,
            attn_mask=self.attn_mask,
            metadata=self.metadata,
            softmax_scale=self.softmax_scale,
            mask_mode=self.mask_mode,
            win_left=self.win_left,
            win_right=self.win_right,
            max_seqlen_q=self.max_seqlen_q,
            max_seqlen_kv=self.max_seqlen_kv,
            layout_q=self.layout_q,
            layout_q_descale=self.layout_q_descale,
            layout_kv=self.layout_kv,
            layout_out=self.layout_out,
            return_softmax_lse=self.return_softmax_lse,
        )

        if not self.return_softmax_lse:
            if isinstance(lse_out, torch.Tensor):
                lse_out = lse_out.reshape(0)
        elif isinstance(lse_out, torch.Tensor) and lse_out.ndim == 2:
            lse_out = lse_out.transpose(0, 1).contiguous()
        return atten_out, lse_out
