# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""PyTorch registration for GenericBlockSparseAttentionGrad."""

from enum import IntEnum
from typing import List, Optional, Tuple, Union

import torch
from torch.library import impl

from cann_ops_transformer.op_builder import OpBuilder, get_as_library

GSAG_TASK_LIST_OFFSET = 80  # 8 + 2 * 36
GSAG_TASK_ENTRY_SIZE = 4
GSAG_METADATA_OP_NAME = "generic_block_sparse_attention_grad_metadata"


class MaskMode(IntEnum):
    """mask_mode 枚举：对外 str/int/IntEnum 经由本枚举映射为传给算子侧的 int。"""

    CAUSAL = 1


def _resolve_mask_mode(mask_mode: Union[str, int, MaskMode, None]) -> int:
    """对外 str/int/IntEnum/None mask_mode 统一为传给算子侧的 int。

    None 视为默认 CAUSAL（当前唯一支持取值）。
    """
    if mask_mode is None:
        return int(MaskMode.CAUSAL)
    if isinstance(mask_mode, str):
        try:
            return int(MaskMode[mask_mode.strip().upper()])
        except KeyError as exc:
            valid = ", ".join(m.name.lower() for m in MaskMode)
            raise ValueError(
                f"mask_mode should be one of [{valid}], but got {mask_mode!r}"
            ) from exc
    try:
        return int(MaskMode(mask_mode))
    except ValueError as exc:
        valid = ", ".join(f"{m.name}={m.value}" for m in MaskMode)
        raise ValueError(
            f"mask_mode should be one of [{valid}], but got {mask_mode!r}"
        ) from exc


def calc_gsag_metadata_size(batch_size: int, num_heads_q: int, num_j: int) -> int:
    """Required metadata int32 length: TASK_LIST_OFFSET + B * N1 * J * TASK_ENTRY_SIZE."""
    return (
        GSAG_TASK_LIST_OFFSET + batch_size * num_heads_q * num_j * GSAG_TASK_ENTRY_SIZE
    )


def _max_segment_from_cu_seqlens(cu_seqlens: torch.Tensor) -> int:
    if cu_seqlens.numel() < 2:
        raise ValueError("cu_seqlens must have at least 2 elements to infer max_seqlen")
    diffs = cu_seqlens[1:] - cu_seqlens[:-1]
    return int(diffs.max().item())


def _resolve_max_seqlen_q(
    max_seqlen_q: Optional[int],
    *,
    sparse_block_idx: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor],
    seqused_q: Optional[torch.Tensor],
) -> int:
    """Resolve max_seqlen_q for aclnn (must be >=0). None → infer from seqused/cu/sparse."""
    if max_seqlen_q is not None:
        value = int(max_seqlen_q)
        if value < 0:
            raise ValueError(f"max_seqlen_q must be >= 0, but got {value}")
        return value
    if seqused_q is not None:
        return int(seqused_q.max().item())
    if cu_seqlens_q is not None:
        return _max_segment_from_cu_seqlens(cu_seqlens_q)
    if sparse_block_idx.dim() == 4:
        return int(sparse_block_idx.shape[3])
    raise ValueError(
        "cannot infer max_seqlen_q; pass max_seqlen_q explicitly or provide "
        "seqused_q / cu_seqlens_q / 4D sparse_block_idx"
    )


def _resolve_max_seqlen_kv(
    max_seqlen_kv: Optional[int],
    *,
    sparse_block_idx: torch.Tensor,
    block_shape: List[int],
    cu_seqlens_kv: Optional[torch.Tensor],
    seqused_kv: Optional[torch.Tensor],
) -> int:
    """Resolve max_seqlen_kv for aclnn (must be >=0). None → infer from seqused/cu/J*block_y."""
    if max_seqlen_kv is not None:
        value = int(max_seqlen_kv)
        if value < 0:
            raise ValueError(f"max_seqlen_kv must be >= 0, but got {value}")
        return value
    if seqused_kv is not None:
        return int(seqused_kv.max().item())
    if cu_seqlens_kv is not None:
        return _max_segment_from_cu_seqlens(cu_seqlens_kv)
    if sparse_block_idx.dim() == 4 and len(block_shape) >= 2:
        num_j = int(sparse_block_idx.shape[2])
        block_y = int(block_shape[1])
        if num_j > 0 and block_y > 0:
            # Satisfies host check: J == ceil(max_seqlen_kv / block_y)
            return num_j * block_y
    raise ValueError(
        "cannot infer max_seqlen_kv; pass max_seqlen_kv explicitly or provide "
        "seqused_kv / cu_seqlens_kv / 4D sparse_block_idx with block_shape"
    )


class GenericBlockSparseAttentionGradOpBuilder(OpBuilder):
    def __init__(self):
        super().__init__("generic_block_sparse_attention_grad", category="attention")

    def sources(self):
        """Path to C++ source code (packaged as csrc/attention/*.cpp)."""
        return ["csrc/attention/generic_block_sparse_attention_grad.cpp"]

    def schema(self):
        return [
            "generic_block_sparse_attention_grad_metadata("
            "Tensor sparse_block_idx, Tensor sparse_block_count, "
            "int num_heads_q, int num_heads_kv, int head_dim, int[] block_shape, *, "
            "Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_kv=None, "
            "Tensor? seqused_q=None, Tensor? seqused_kv=None, "
            "int? max_seqlen_q=None, int? max_seqlen_kv=None, bool is_packed_gqa=True, "
            'str layout_q="TND", str layout_kv="TND", '
            "int mask_mode=1, int softmax_precision=0, "
            "int win_left=-1, int win_right=-1) -> Tensor",
            "generic_block_sparse_attention_grad("
            "Tensor q, Tensor k, Tensor v, Tensor dout, Tensor attn_out, Tensor softmax_lse, "
            "Tensor sparse_block_idx, Tensor sparse_block_count, int[] block_shape, *, "
            "Tensor? metadata=None, Tensor? attn_mask=None, Tensor? cu_seqlens_q=None, "
            "Tensor? cu_seqlens_kv=None, Tensor? seqused_q=None, Tensor? seqused_kv=None, "
            "bool is_packed_gqa=True, "
            'str layout_q="TND", str layout_kv="TND", '
            "float softmax_scale=1.0, int mask_mode=1, int softmax_precision=0, "
            "int win_left=-1, int win_right=-1) -> (Tensor, Tensor, Tensor)",
        ]

    def register_meta(self):
        @torch.library.register_fake("cann_ops_transformer::" + GSAG_METADATA_OP_NAME)
        def generic_block_sparse_attention_grad_metadata_meta(
            sparse_block_idx: torch.Tensor,
            sparse_block_count: torch.Tensor,
            num_heads_q: int,
            num_heads_kv: int,
            head_dim: int,
            block_shape: List[int],
            *,
            cu_seqlens_q: Optional[torch.Tensor] = None,
            cu_seqlens_kv: Optional[torch.Tensor] = None,
            seqused_q: Optional[torch.Tensor] = None,
            seqused_kv: Optional[torch.Tensor] = None,
            max_seqlen_q: Optional[int] = None,
            max_seqlen_kv: Optional[int] = None,
            is_packed_gqa: bool = True,
            layout_q: str = "TND",
            layout_kv: str = "TND",
            mask_mode: int = 1,
            softmax_precision: int = 0,
            win_left: int = -1,
            win_right: int = -1,
        ) -> torch.Tensor:
            del (
                sparse_block_count,
                num_heads_kv,
                head_dim,
                is_packed_gqa,
                layout_q,
                layout_kv,
                mask_mode,
                softmax_precision,
                win_left,
                win_right,
            )
            if len(block_shape) < 2:
                raise ValueError(
                    "block_shape is required and must be [block_x, block_y]"
                )
            block_y = int(block_shape[1])
            batch = int(sparse_block_idx.shape[0])
            num_j = int(sparse_block_idx.shape[2]) if sparse_block_idx.dim() == 4 else 0
            resolved_max_kv = _resolve_max_seqlen_kv(
                max_seqlen_kv,
                sparse_block_idx=sparse_block_idx,
                block_shape=block_shape,
                cu_seqlens_kv=cu_seqlens_kv,
                seqused_kv=seqused_kv,
            )
            # touch q resolver for consistent validation in fake path
            _resolve_max_seqlen_q(
                max_seqlen_q,
                sparse_block_idx=sparse_block_idx,
                cu_seqlens_q=cu_seqlens_q,
                seqused_q=seqused_q,
            )
            if num_j <= 0 and resolved_max_kv > 0:
                num_j = (resolved_max_kv + block_y - 1) // block_y
            meta_size = calc_gsag_metadata_size(batch, int(num_heads_q), max(num_j, 0))
            return torch.empty(
                (meta_size,), dtype=torch.int32, device=sparse_block_idx.device
            )

        @torch.library.register_fake(
            "cann_ops_transformer::generic_block_sparse_attention_grad"
        )
        def generic_block_sparse_attention_grad_meta(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            dout: torch.Tensor,
            attn_out: torch.Tensor,
            softmax_lse: torch.Tensor,
            sparse_block_idx: torch.Tensor,
            sparse_block_count: torch.Tensor,
            block_shape: List[int],
            *,
            metadata: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
            cu_seqlens_q: Optional[torch.Tensor] = None,
            cu_seqlens_kv: Optional[torch.Tensor] = None,
            seqused_q: Optional[torch.Tensor] = None,
            seqused_kv: Optional[torch.Tensor] = None,
            is_packed_gqa: bool = True,
            layout_q: str = "TND",
            layout_kv: str = "TND",
            softmax_scale: float = 1.0,
            mask_mode: int = 1,
            softmax_precision: int = 0,
            win_left: int = -1,
            win_right: int = -1,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            del (
                dout,
                attn_out,
                softmax_lse,
                sparse_block_idx,
                sparse_block_count,
                block_shape,
                metadata,
                attn_mask,
                cu_seqlens_q,
                cu_seqlens_kv,
                seqused_q,
                seqused_kv,
                is_packed_gqa,
                layout_q,
                layout_kv,
                softmax_scale,
                mask_mode,
                softmax_precision,
                win_left,
                win_right,
            )
            return (
                torch.empty_like(q, device="meta"),
                torch.empty_like(k, device="meta"),
                torch.empty_like(v, device="meta"),
            )


generic_block_sparse_attention_grad_op_builder = (
    GenericBlockSparseAttentionGradOpBuilder()
)
generic_block_sparse_attention_grad_op_builder._ensure_initialized()


@impl(get_as_library(), GSAG_METADATA_OP_NAME, "PrivateUse1")
def generic_block_sparse_attention_grad_metadata(
    sparse_block_idx: torch.Tensor,
    sparse_block_count: torch.Tensor,
    num_heads_q: int,
    num_heads_kv: int,
    head_dim: int,
    block_shape: List[int],
    *,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_kv: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_kv: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_kv: Optional[int] = None,
    is_packed_gqa: bool = True,
    layout_q: str = "TND",
    layout_kv: str = "TND",
    mask_mode: Union[int, MaskMode, str] = MaskMode.CAUSAL,
    softmax_precision: int = 0,
    win_left: int = -1,
    win_right: int = -1,
) -> torch.Tensor:
    if len(block_shape) < 2:
        raise ValueError("block_shape is required and must be [block_x, block_y]")
    max_seqlen_q_i = _resolve_max_seqlen_q(
        max_seqlen_q,
        sparse_block_idx=sparse_block_idx,
        cu_seqlens_q=cu_seqlens_q,
        seqused_q=seqused_q,
    )
    max_seqlen_kv_i = _resolve_max_seqlen_kv(
        max_seqlen_kv,
        sparse_block_idx=sparse_block_idx,
        block_shape=block_shape,
        cu_seqlens_kv=cu_seqlens_kv,
        seqused_kv=seqused_kv,
    )
    mask_mode_i = _resolve_mask_mode(mask_mode)
    op_module = generic_block_sparse_attention_grad_op_builder.load()
    return op_module.generic_block_sparse_attention_grad_metadata(
        sparse_block_idx,
        sparse_block_count,
        cu_seqlens_q,
        cu_seqlens_kv,
        seqused_q,
        seqused_kv,
        max_seqlen_q_i,
        max_seqlen_kv_i,
        num_heads_q,
        num_heads_kv,
        head_dim,
        block_shape,
        int(is_packed_gqa),
        layout_q,
        layout_kv,
        mask_mode_i,
        softmax_precision,
        win_left,
        win_right,
    )


@impl(get_as_library(), "generic_block_sparse_attention_grad", "PrivateUse1")
def generic_block_sparse_attention_grad(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dout: torch.Tensor,
    attn_out: torch.Tensor,
    softmax_lse: torch.Tensor,
    sparse_block_idx: torch.Tensor,
    sparse_block_count: torch.Tensor,
    block_shape: List[int],
    *,
    metadata: Optional[torch.Tensor] = None,
    attn_mask: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_kv: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_kv: Optional[torch.Tensor] = None,
    is_packed_gqa: bool = True,
    layout_q: str = "TND",
    layout_kv: str = "TND",
    softmax_scale: float = 1.0,
    mask_mode: Union[int, MaskMode, str] = MaskMode.CAUSAL,
    softmax_precision: int = 0,
    win_left: int = -1,
    win_right: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(block_shape) < 2:
        raise ValueError("block_shape is required and must be [block_x, block_y]")
    mask_mode_i = _resolve_mask_mode(mask_mode)
    op_module = generic_block_sparse_attention_grad_op_builder.load()
    return op_module.generic_block_sparse_attention_grad(
        q,
        k,
        v,
        dout,
        attn_out,
        softmax_lse,
        sparse_block_idx,
        sparse_block_count,
        metadata,
        attn_mask,
        cu_seqlens_q,
        cu_seqlens_kv,
        seqused_q,
        seqused_kv,
        block_shape,
        int(is_packed_gqa),
        layout_q,
        layout_kv,
        softmax_scale,
        mask_mode_i,
        softmax_precision,
        win_left,
        win_right,
    )


@torch.library.register_kernel(
    "cann_ops_transformer::generic_block_sparse_attention_grad", None
)
def generic_block_sparse_attention_grad_fallback(*args, **kwargs):
    return generic_block_sparse_attention_grad(*args, **kwargs)


@torch.library.register_kernel("cann_ops_transformer::" + GSAG_METADATA_OP_NAME, None)
def generic_block_sparse_attention_grad_metadata_fallback(*args, **kwargs):
    return generic_block_sparse_attention_grad_metadata(*args, **kwargs)


torch.compiler.allow_in_graph(generic_block_sparse_attention_grad_metadata)
torch.compiler.allow_in_graph(generic_block_sparse_attention_grad)
