#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""TTK metadata-first adapter for the installed quant_block_sparse_attn API."""

import math
import random
import sys
from pathlib import Path
from typing import List, Optional

import torch

_CUSTOM_OPS_LOADED = False
_FP32_BYTES = 4


# ==================================================================================================
# Common helpers and public adapters
# ==================================================================================================


def _ensure_cann_ops():
    global _CUSTOM_OPS_LOADED
    if not _CUSTOM_OPS_LOADED:
        pytest_dir = str(Path(__file__).resolve().parents[1] / "pytest")
        if pytest_dir not in sys.path:
            sys.path.insert(0, pytest_dir)
        import custom_ops  # noqa: F401 - loads .so + registers torch.ops.custom namespace

        _CUSTOM_OPS_LOADED = True


def _synchronize_npu(tensor):
    """Finish eager preparation before ACLGraph starts capturing."""
    if tensor.device.type != "npu":
        return
    import torch_npu

    torch_npu.npu.synchronize()


def _none_if_empty(tensor):
    return None if tensor is None or int(tensor.numel()) == 0 else tensor


class QuantBlockSparseAttnMetadataBuilder:
    """Derive and create the metadata consumed by the main operator."""

    @staticmethod
    def max_seq(cu_seqlens, seqused, fallback):
        if seqused is not None and seqused.numel() > 0:
            return int(seqused.detach().cpu().max().item())
        if cu_seqlens is not None and cu_seqlens.numel() > 1:
            values = cu_seqlens.detach().cpu()
            return int((values[1:] - values[:-1]).max().item())
        return int(fallback)

    @staticmethod
    def batch_size(q, layout_q, cu_seqlens_q, seqused_q, sparse_indices=None):
        if layout_q == "BSND":
            return int(q.shape[0])
        if seqused_q is not None:
            return int(seqused_q.numel())
        if cu_seqlens_q is not None:
            return max(int(cu_seqlens_q.numel()) - 1, 0)
        if sparse_indices is not None:
            return int(sparse_indices.shape[0])
        return 1

    @classmethod
    def build(
        cls,
        query,
        key,
        value,
        q_descale,
        k_descale,
        v_descale,
        p_scale,
        sparse_indices,
        sparse_seq_len,
        atten_mask,
        *,
        cu_seqlens_q=None,
        cu_seqlens_kv=None,
        seqused_q=None,
        seqused_kv=None,
        block_table=None,
        softmax_scale=1.0,
        sparse_q_block_size=128,
        sparse_kv_block_size=128,
        layout_q="BSND",
        layout_kv="PA_BNBD",
        layout_sparse_indices="B_N_Qb_Kb",
        mask_mode=3,
        quant_mode=1,
        max_seqlen_q=0,
        max_seqlen_kv=0,
    ):
        _ensure_cann_ops()
        if layout_q == "BSND":
            num_heads_q = int(query.shape[2])
            q_fallback = int(query.shape[1])
        elif layout_q == "NTD":
            num_heads_q = int(query.shape[0])
            q_fallback = int(query.shape[1])
        else:
            num_heads_q = int(query.shape[1])
            q_fallback = int(query.shape[0])
        num_heads_kv = int(key.shape[1])
        kv_fallback = (
            int(key.shape[0] * key.shape[2]) if key.dim() == 4 else int(key.shape[0])
        )
        metadata = torch.ops.custom.npu_quant_block_sparse_attn_metadata(
            sparse_seq_len,
            num_heads_q,
            num_heads_kv,
            int(query.shape[-1]),
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=_none_if_empty(cu_seqlens_kv),
            seqused_q=_none_if_empty(seqused_q),
            seqused_kv=seqused_kv,
            batch_size=cls.batch_size(
                query, layout_q, cu_seqlens_q, seqused_q, sparse_indices
            ),
            sparse_block_size_q=int(sparse_q_block_size),
            sparse_block_size_k=int(sparse_kv_block_size),
            quant_mode=int(quant_mode),
            mask_mode=int(mask_mode),
            layout_q=layout_q,
            layout_kv=layout_kv,
            layout_sparse_indices=layout_sparse_indices,
        )
        if hasattr(metadata, "to") and metadata.device != query.device:
            metadata = metadata.to(query.device)
        return metadata


def _prepare_operator_inputs(
    query,
    key,
    value,
    q_descale,
    k_descale,
    v_descale,
    p_scale,
    sparse_indices,
    sparse_seq_len,
    atten_mask,
    cu_seqlens_q,
    cu_seqlens_kv,
    seqused_q,
    seqused_kv,
    block_table,
    metadata,
    *,
    quant_mode,
    softmax_scale,
    mask_mode,
    sparse_block_size_q,
    sparse_block_size_kv,
    layout_q,
    layout_kv,
    layout_sparse_indices,
):
    p_scale = _none_if_empty(p_scale)
    if metadata is None or int(metadata.numel()) < 8:
        metadata = QuantBlockSparseAttnMetadataBuilder.build(
            query,
            key,
            value,
            q_descale,
            k_descale,
            v_descale,
            p_scale,
            sparse_indices,
            sparse_seq_len,
            atten_mask,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=_none_if_empty(cu_seqlens_kv),
            seqused_q=_none_if_empty(seqused_q),
            seqused_kv=seqused_kv,
            block_table=block_table,
            softmax_scale=softmax_scale,
            sparse_q_block_size=sparse_block_size_q,
            sparse_kv_block_size=sparse_block_size_kv,
            layout_q=layout_q,
            layout_kv=layout_kv,
            layout_sparse_indices=layout_sparse_indices,
            mask_mode=mask_mode,
            quant_mode=quant_mode,
        )
    if int(quant_mode) == 1 and _should_pack_combined_kv():
        key, value, k_descale = _fp8_pack_combined_kv(key, value, k_descale)
    return key, value, k_descale, p_scale, metadata


class QuantBlockSparseAttnGraph(torch.nn.Module):
    """Build metadata before ACLGraph capture and capture only the main operator."""

    def __init__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        q_descale: torch.Tensor,
        k_descale: torch.Tensor,
        v_descale: torch.Tensor,
        sparse_indices: torch.Tensor,
        sparse_seq_len: torch.Tensor,
        p_scale: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_kv: torch.Tensor,
        seqused_q: torch.Tensor,
        seqused_kv: torch.Tensor,
        block_table: torch.Tensor,
        atten_mask: torch.Tensor,
        metadata: torch.Tensor,
        *,
        quant_mode: int = 1,
        softmax_scale: float = 1.0,
        mask_mode: int = 3,
        blocksize: int = 0,
        sparse_block_size_q: int = 128,
        sparse_block_size_kv: int = 128,
        layout_q: str = "TND",
        layout_kv: str = "PA_BNBD",
        layout_out: str = "TND",
        layout_sparse_indices: str = "B_N_Qb_Kb",
        return_softmax_lse: bool = False,
        quant_matmul: bool = False,
        batch_size: int = 0,
        num_heads_q: int = 0,
        num_heads_kv: int = 0,
        head_dim: int = 0,
    ):
        super().__init__()
        del quant_matmul  # MXFP8 golden-only switch; never forward it to the operator.
        _ensure_cann_ops()
        _synchronize_npu(query)
        key, value, k_descale, p_scale, metadata = _prepare_operator_inputs(
            query,
            key,
            value,
            q_descale,
            k_descale,
            v_descale,
            p_scale,
            sparse_indices,
            sparse_seq_len,
            atten_mask,
            cu_seqlens_q,
            cu_seqlens_kv,
            seqused_q,
            seqused_kv,
            block_table,
            metadata,
            quant_mode=quant_mode,
            softmax_scale=softmax_scale,
            mask_mode=mask_mode,
            sparse_block_size_q=sparse_block_size_q,
            sparse_block_size_kv=sparse_block_size_kv,
            layout_q=layout_q,
            layout_kv=layout_kv,
            layout_sparse_indices=layout_sparse_indices,
        )
        _synchronize_npu(query)
        for name, tensor in (
            ("query", query),
            ("key", key),
            ("value", value),
            ("q_descale", q_descale),
            ("k_descale", k_descale),
            ("v_descale", v_descale),
            ("p_scale", p_scale),
            ("sparse_indices", sparse_indices),
            ("sparse_seq_len", sparse_seq_len),
            ("cu_seqlens_q", cu_seqlens_q),
            ("cu_seqlens_kv", _none_if_empty(cu_seqlens_kv)),
            ("seqused_q", _none_if_empty(seqused_q)),
            ("seqused_kv", seqused_kv),
            ("block_table", block_table),
            ("atten_mask", atten_mask),
            ("metadata", metadata),
        ):
            self.register_buffer(name, tensor, persistent=False)
        self.softmax_scale = float(softmax_scale)
        self.sparse_block_size_q = int(sparse_block_size_q)
        self.sparse_block_size_kv = int(sparse_block_size_kv)
        self.layout_q = layout_q
        self.layout_kv = layout_kv
        self.layout_out = layout_out
        self.layout_sparse_indices = layout_sparse_indices
        self.quant_mode = int(quant_mode)
        self.mask_mode = int(mask_mode)
        self.return_softmax_lse = bool(return_softmax_lse)

    def forward(self):
        return torch.ops.custom.npu_quant_block_sparse_attn(
            self.query,
            self.key,
            self.value,
            self.q_descale,
            self.k_descale,
            self.v_descale,
            self.p_scale,
            self.sparse_indices,
            self.sparse_seq_len,
            self.atten_mask,
            cu_seqlens_q=self.cu_seqlens_q,
            cu_seqlens_kv=self.cu_seqlens_kv,
            seqused_q=self.seqused_q,
            seqused_kv=self.seqused_kv,
            block_table=self.block_table,
            metadata=self.metadata,
            softmax_scale=self.softmax_scale,
            sparse_q_block_size=self.sparse_block_size_q,
            sparse_kv_block_size=self.sparse_block_size_kv,
            layout_kv=self.layout_kv,
            layout_q=self.layout_q,
            layout_sparse_indices=self.layout_sparse_indices,
            layout_out=self.layout_out,
            quant_mode=self.quant_mode,
            mask_mode=self.mask_mode,
            return_softmax_lse=self.return_softmax_lse,
        )


def quant_block_sparse_attn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    q_descale: torch.Tensor,
    k_descale: torch.Tensor,
    v_descale: torch.Tensor,
    sparse_indices: torch.Tensor,
    sparse_seq_len: torch.Tensor,
    p_scale: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    seqused_q: torch.Tensor,
    seqused_kv: torch.Tensor,
    block_table: torch.Tensor,
    atten_mask: torch.Tensor,
    metadata: torch.Tensor,
    *,
    quant_mode: int = 1,
    softmax_scale: float = 1.0,
    mask_mode: int = 3,
    blocksize: int = 0,
    sparse_block_size_q: int = 128,
    sparse_block_size_kv: int = 128,
    layout_q: str = "TND",
    layout_kv: str = "PA_BNBD",
    layout_out: str = "TND",
    layout_sparse_indices: str = "B_N_Qb_Kb",
    return_softmax_lse: bool = False,
    quant_matmul: bool = False,
    batch_size: int = 0,
    num_heads_q: int = 0,
    num_heads_kv: int = 0,
    head_dim: int = 0,
):
    """Adapt the shared CSV inputs and call the installed QBSA operator."""
    del quant_matmul  # MXFP8 golden-only switch; never forward it to the operator.
    _ensure_cann_ops()
    key, value, k_descale, p_scale, metadata = _prepare_operator_inputs(
        query,
        key,
        value,
        q_descale,
        k_descale,
        v_descale,
        p_scale,
        sparse_indices,
        sparse_seq_len,
        atten_mask,
        cu_seqlens_q,
        cu_seqlens_kv,
        seqused_q,
        seqused_kv,
        block_table,
        metadata,
        quant_mode=quant_mode,
        softmax_scale=softmax_scale,
        mask_mode=mask_mode,
        sparse_block_size_q=sparse_block_size_q,
        sparse_block_size_kv=sparse_block_size_kv,
        layout_q=layout_q,
        layout_kv=layout_kv,
        layout_sparse_indices=layout_sparse_indices,
    )
    return torch.ops.custom.npu_quant_block_sparse_attn(
        query,
        key,
        value,
        q_descale,
        k_descale,
        v_descale,
        p_scale,
        sparse_indices,
        sparse_seq_len,
        atten_mask,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_kv=_none_if_empty(cu_seqlens_kv),
        seqused_q=_none_if_empty(seqused_q),
        seqused_kv=seqused_kv,
        block_table=block_table,
        metadata=metadata,
        softmax_scale=float(softmax_scale),
        sparse_q_block_size=int(sparse_block_size_q),
        sparse_kv_block_size=int(sparse_block_size_kv),
        layout_kv=layout_kv,
        layout_q=layout_q,
        layout_sparse_indices=layout_sparse_indices,
        layout_out=layout_out,
        quant_mode=int(quant_mode),
        mask_mode=int(mask_mode),
        return_softmax_lse=bool(return_softmax_lse),
    )


# ==================================================================================================
# FP8-only helpers
# ==================================================================================================
def _should_pack_combined_kv():
    """Skip combined-KV packing only for TTK E2E without --plugin."""
    try:
        from ttk.utilities import get_global_storage
    except (ImportError, ModuleNotFoundError):
        # pytest、直接调用或未安装 TTK：维持原来的正常 pack 行为
        return True

    switches = get_global_storage()

    # 非 TTK E2E 环境保持原行为
    if getattr(switches, "test_mode", None) != "framework-api":
        return True

    # TTK E2E：
    # 带 --plugin    -> plugin_path 非空 -> 正常 pack
    # 不带 --plugin  -> plugin_path 为空 -> 负向用例不 pack
    return bool(getattr(switches, "plugin_path", None))


def _fp8_pack_combined_kv(key, value, k_descale):
    """Pack separate PA key/value/k_descale into a combined KV storage.

    Layout per physical block: [key_segment (FP8)] [value_segment (FP8)] [k_scale_segment (FP32)]
    Returns 4D PA views whose shape matches the operator interface and whose block stride
    points at the full physical KV-cache block.
    """
    num_blocks, n2, block_size, d = (
        int(key.shape[0]),
        int(key.shape[1]),
        int(key.shape[2]),
        int(key.shape[3]),
    )
    dv = int(value.shape[-1])

    key_seg = n2 * block_size * d
    value_seg = n2 * block_size * dv
    k_scale_seg = n2 * block_size * _FP32_BYTES
    pa_block_stride = key_seg + value_seg + k_scale_seg

    storage = torch.zeros(
        (num_blocks * pa_block_stride,), dtype=torch.uint8, device=key.device
    )
    fp8_storage = storage.view(torch.float8_e4m3fn)
    fp32_storage = storage.view(torch.float32)

    key_view = torch.as_strided(
        fp8_storage, key.shape, (pa_block_stride, block_size * d, d, 1), 0
    )
    value_view = torch.as_strided(
        fp8_storage, value.shape, (pa_block_stride, block_size * dv, dv, 1), key_seg
    )
    k_scale_shape = (
        tuple(k_descale.shape)
        if k_descale.dim() == 4
        else tuple(k_descale.shape) + (1,)
    )
    k_scale_view = torch.as_strided(
        fp32_storage,
        k_scale_shape,
        (pa_block_stride // _FP32_BYTES, block_size, 1, 1),
        (key_seg + value_seg) // _FP32_BYTES,
    )

    key_view.copy_(key)
    value_view.copy_(value)
    k_scale_view.copy_(k_descale if k_descale.dim() == 4 else k_descale.unsqueeze(-1))

    return key_view, value_view, k_scale_view
