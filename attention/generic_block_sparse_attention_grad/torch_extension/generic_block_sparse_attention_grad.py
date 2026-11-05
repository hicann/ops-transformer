# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""PyTorch registration for GenericBlockSparseAttentionGrad."""

import math
from enum import IntEnum
from typing import List, Optional, Tuple, Union

import torch
from torch.library import impl

from cann_ops_transformer.op_builder import OpBuilder, get_as_library

GSAG_TASK_LIST_OFFSET = 80  # 8 + 2 * 36
GSAG_TASK_ENTRY_SIZE = 4
GSAG_METADATA_OP_NAME = "generic_block_sparse_attention_grad_metadata"
ARC22_GBSAG_TASK_LIST_SIZE = 198  # 6 + 64 * 3（arch22 ABI 上界，具名常量）
GSAG_ARCH35_SOC_MARK = "Ascend950"  # 与 csrc / AICPU 分发器保持同一判定串
FLOAT32_MAX = torch.finfo(torch.float32).max
_OP_PREFIX = "GenericBlockSparseAttentionGrad"


class MaskMode(IntEnum):
    """mask_mode 枚举：对外 str/int/IntEnum 经由本枚举映射为传给算子侧的 int。

    QSAG（layout_sparse_pattern==0）支持 mask_mode 0（无掩码）/ 1（CAUSAL）。
    """

    NO_MASK = 0
    CAUSAL = 1


class ResidualBlockMode(IntEnum):
    """尾部不完整 KV 块的处理模式。"""

    MARKED_BY_SPARSE_BLK_IDX = 0  # 是否参与计算由 sparse_block_idx 决定。
    INCOMPLETE_BLK_KEPT_BUT_NOT_IN_SPARSE_BLK_IDX = (
        1  # 必定参与计算但不在索引中；QSAG（layout_sparse_pattern=0）使用。
    )


def _resolve_mask_mode(mask_mode: Union[str, int, MaskMode, None]) -> int:
    """对外 str/int/IntEnum/None mask_mode 统一为传给算子侧的 int。

    None 视为默认 CAUSAL。QSAG（layout_sparse_pattern==0）支持 0 (NO_MASK) / 1 (CAUSAL)。
    """
    if mask_mode is None:
        return int(MaskMode.CAUSAL)
    if isinstance(mask_mode, str):
        try:
            return int(MaskMode[mask_mode.strip().upper()])
        except KeyError as exc:
            valid = ", ".join(f"{m.name}={m.value}" for m in MaskMode)
            raise ValueError(
                f"{_OP_PREFIX}: only support mask_mode in [{valid}], got {mask_mode!r}"
            ) from exc
    try:
        return int(MaskMode(mask_mode))
    except ValueError as exc:
        valid = ", ".join(f"{m.name}={m.value}" for m in MaskMode)
        raise ValueError(
            f"{_OP_PREFIX}: only support mask_mode in [{valid}], got {mask_mode!r}"
        ) from exc


def _resolve_residual_block_mode(
    residual_block_mode: Union[str, int, ResidualBlockMode],
) -> int:
    """将枚举名字符串、整数或枚举统一为算子侧整数；支持范围由属性检查负责。"""
    try:
        if isinstance(residual_block_mode, str):
            return int(ResidualBlockMode[residual_block_mode.strip().upper()])
        return int(ResidualBlockMode(residual_block_mode))
    except (KeyError, TypeError, ValueError) as exc:
        valid = ", ".join(f"{m.name}={m.value}" for m in ResidualBlockMode)
        raise ValueError(
            f"{_OP_PREFIX}: invalid residual_block_mode {residual_block_mode!r}; "
            f"expected an enum name or value in [{valid}]. Supported: "
            "0 (MARKED_BY_SPARSE_BLK_IDX, GBSAG) / 1 "
            "(INCOMPLETE_BLK_KEPT_BUT_NOT_IN_SPARSE_BLK_IDX, QSAG)."
        ) from exc


def _check_sparse_attributes(
    layout_sparse_pattern: int, residual_block_mode: int
) -> None:
    if layout_sparse_pattern not in (0, 1):
        raise ValueError(
            f"{_OP_PREFIX}: layout_sparse_pattern currently only supports "
            "0 (QSAG) or 1 (BNKQ/GBSAG), got {layout_sparse_pattern}"
        )
    if layout_sparse_pattern == 0 and residual_block_mode != (
        ResidualBlockMode.INCOMPLETE_BLK_KEPT_BUT_NOT_IN_SPARSE_BLK_IDX
    ):
        raise ValueError(
            f"{_OP_PREFIX}: layout_sparse_pattern==0 (QSAG) requires "
            "residual_block_mode == "
            "ResidualBlockMode.INCOMPLETE_BLK_KEPT_BUT_NOT_IN_SPARSE_BLK_IDX (1), "
            f"got {residual_block_mode}"
        )
    if layout_sparse_pattern == 1 and residual_block_mode != (
        ResidualBlockMode.MARKED_BY_SPARSE_BLK_IDX
    ):
        raise ValueError(
            f"{_OP_PREFIX}: layout_sparse_pattern==1 (GBSAG) requires "
            "residual_block_mode == "
            "ResidualBlockMode.MARKED_BY_SPARSE_BLK_IDX (0), "
            f"got {residual_block_mode}"
        )


def _check_qsag(
    q,
    k,
    v,
    dout,
    attn_out,
    lse,
    indices,
    counts,
    block_shape,
    *,
    metadata,
    attn_mask,
    cu_q,
    cu_kv,
    used_q,
    used_kv,
    layout_q,
    layout_kv,
    residual_block_mode,
    is_consistent_topk,
    scale,
    mask,
    precision,
    left,
    right,
):
    """Descriptor checks only: preserve asynchronous execution and FakeTensor support."""
    if residual_block_mode != 1:
        raise ValueError("QSAG requires residual_block_mode=1")
    if layout_q != layout_kv or layout_q not in ("TND", "BNSD", "BSND"):
        raise ValueError("QSAG requires matching TND/BNSD/BSND layouts")
    if (
        mask not in (0, 1)
        or precision != 0
        or left != -1
        or right != -1
        or not math.isfinite(scale)
        or abs(scale) > FLOAT32_MAX
    ):
        raise ValueError(
            "QSAG supports mask 0/1, precision 0, windows -1 and finite FP32 scale"
        )
    if metadata is not None or attn_mask is not None:
        raise ValueError("QSAG requires metadata=None and attn_mask=None")
    packed = layout_q == "TND"
    rank = 3 if packed else 4
    if q.dim() != rank or k.dim() != rank:
        raise ValueError("Q/K ranks must match layout")
    if v.shape != k.shape or dout.shape != q.shape or attn_out.shape != q.shape:
        raise ValueError("Q/dout/attn_out and K/V shapes must match")
    if q.dtype not in (torch.float16, torch.bfloat16) or any(
        t.dtype != q.dtype for t in (k, v, dout, attn_out)
    ):
        raise ValueError("QSAG inputs must have matching FP16/BF16 dtype")
    if (
        lse.dtype != torch.float32
        or indices.dtype != torch.int32
        or counts.dtype != torch.int32
    ):
        raise ValueError("QSAG requires FP32 LSE and INT32 indices/counts")
    if lse.shape not in (q.shape[:-1], q.shape[:-1] + (1,)):
        raise ValueError(
            "QSAG LSE must follow Q layout with D removed or replaced by 1"
        )
    h = 1 if packed or layout_q == "BNSD" else 2
    sq_axis = 0 if packed else (2 if layout_q == "BNSD" else 1)
    nq, nk, d = q.shape[h], k.shape[h], q.shape[-1]
    if nq <= 0 or nk <= 0 or nq % nk or d <= 0 or k.shape[-1] != d:
        raise ValueError(
            "QSAG requires positive divisible head counts and common positive D"
        )
    # Architecture-specific Bs/G/D limits are checked by the selected tiling.
    if len(block_shape) != 2 or block_shape[0] != 1 or block_shape[1] <= 0:
        raise ValueError("QSAG requires block_shape=[1,Bs], Bs>0")
    if indices.dim() != 4 or counts.dim() != 3:
        raise ValueError("QSAG indices/count ranks must be 4/3")
    b, n, qcap, topk = indices.shape
    if (
        b <= 0
        or n != nk
        or not 1 <= topk <= 2147483647
        or counts.shape != (b, nk, qcap)
    ):
        raise ValueError(
            "QSAG requires indices [B,N2,Qcap,topK] and counts [B,N2,Qcap], topK>=1"
        )
    if packed:
        for t in (cu_q, cu_kv):
            if t is None or t.dtype != torch.int64 or t.shape != (b + 1,):
                raise ValueError("TND QSAG requires INT64 cu lengths [B+1]")
        for t in (used_q, used_kv):
            if t is not None and (t.dtype != torch.int32 or t.shape != (b,)):
                raise ValueError("QSAG seqused must be INT32 [B]")
    elif any(t is not None for t in (cu_q, cu_kv, used_q, used_kv)):
        raise ValueError("dense QSAG does not accept cu lengths or seqused")
    elif q.shape[0] != b or k.shape[0] != b or qcap != q.shape[sq_axis]:
        raise ValueError("QSAG dense batch and Qcap must match Q/K")
    for t in (
        q,
        k,
        v,
        dout,
        attn_out,
        lse,
        indices,
        counts,
        cu_q,
        cu_kv,
        used_q,
        used_kv,
    ):
        # Match qsag::ValidShape's bound for padded FP32 workspace offsets.
        if t is not None and t.numel() > (2**63 - 1) // 64:
            raise ValueError("QSAG tensor shape overflow")
        if t is not None and t.device != q.device:
            raise ValueError("all QSAG tensors must be on the same device")


def _is_arch35_soc() -> bool:
    """当前 SoC 是否 arch35（Ascend950）。

    与 csrc CalcGsagMetadataSizeForSoc 的 aclrtGetSocName() + strstr 判定保持一致；
    异常时回退 arch22（固定容量），与 csrc socName == nullptr 的回退一致，
    避免 fake/meta 与真实分配 shape 分叉。
    """
    try:
        return GSAG_ARCH35_SOC_MARK in torch.npu.get_device_name()
    except Exception:
        return False


def calc_gsag_metadata_size(
    batch_size: int, num_heads_q: int, num_j: int, is_arch35: Optional[bool] = None
) -> int:
    """metadata int32 长度（必须与 csrc CalcGsagMetadataSize 逐平台一致）:
    A5/arch35  : TASK_LIST_OFFSET + B * N1 * J * TASK_ENTRY_SIZE
    A2/A3/arch22: ARC22_GBSAG_TASK_LIST_SIZE（6 + 64 * 3 = 198）
    is_arch35 仅供单测显式指定；缺省按当前 SoC 自动判定。
    """
    if is_arch35 is None:
        is_arch35 = _is_arch35_soc()
    if is_arch35:
        return (
            GSAG_TASK_LIST_OFFSET
            + batch_size * num_heads_q * num_j * GSAG_TASK_ENTRY_SIZE
        )
    return ARC22_GBSAG_TASK_LIST_SIZE


def _max_segment_from_cu_seqlens(cu_seqlens: torch.Tensor) -> int:
    if cu_seqlens.numel() < 2:
        raise ValueError(
            f"{_OP_PREFIX}: cu_seqlens must have at least 2 elements to infer max_seqlen"
        )
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
            raise ValueError(
                f"{_OP_PREFIX}: max_seqlen_q must be >= 0, but got {value}"
            )
        return value
    if seqused_q is not None:
        return int(seqused_q.max().item())
    if cu_seqlens_q is not None:
        return _max_segment_from_cu_seqlens(cu_seqlens_q)
    if sparse_block_idx.dim() == 4:
        return int(sparse_block_idx.shape[3])
    raise ValueError(
        f"{_OP_PREFIX}: cannot infer max_seqlen_q; pass max_seqlen_q explicitly or provide "
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
            raise ValueError(
                f"{_OP_PREFIX}: max_seqlen_kv must be >= 0, but got {value}"
            )
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
        f"{_OP_PREFIX}: cannot infer max_seqlen_kv; pass max_seqlen_kv explicitly or provide "
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
            "int? max_seqlen_q=None, int? max_seqlen_kv=None, "
            'str layout_q="TND", str layout_kv="TND", int layout_sparse_pattern=1, '
            "int mask_mode=1, int softmax_precision=0, "
            "int win_left=-1, int win_right=-1, int residual_block_mode=0, bool is_consistent_topk=False) -> Tensor",
            "generic_block_sparse_attention_grad("
            "Tensor q, Tensor k, Tensor v, Tensor dout, Tensor attn_out, Tensor softmax_lse, "
            "Tensor sparse_block_idx, Tensor sparse_block_count, int[] block_shape, *, "
            "Tensor? metadata=None, Tensor? attn_mask=None, Tensor? cu_seqlens_q=None, "
            "Tensor? cu_seqlens_kv=None, Tensor? seqused_q=None, Tensor? seqused_kv=None, "
            'str layout_q="TND", str layout_kv="TND", int layout_sparse_pattern=1, '
            "float softmax_scale=1.0, int mask_mode=1, int softmax_precision=0, "
            "int win_left=-1, int win_right=-1, int residual_block_mode=0, "
            "bool is_consistent_topk=False) -> (Tensor, Tensor, Tensor)",
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
            layout_q: str = "TND",
            layout_kv: str = "TND",
            layout_sparse_pattern: int = 1,
            mask_mode: int = 1,
            softmax_precision: int = 0,
            win_left: int = -1,
            win_right: int = -1,
            residual_block_mode: int = 0,
            is_consistent_topk: bool = False,
        ) -> torch.Tensor:
            _check_sparse_attributes(layout_sparse_pattern, residual_block_mode)
            del (
                is_consistent_topk,
                sparse_block_count,
                num_heads_kv,
                head_dim,
                layout_q,
                layout_kv,
                mask_mode,
                softmax_precision,
                win_left,
                win_right,
            )
            if len(block_shape) != 2:
                raise ValueError(
                    f"{_OP_PREFIX}: block_shape is required and must be [block_x, block_y]"
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
            layout_q: str = "TND",
            layout_kv: str = "TND",
            layout_sparse_pattern: int = 1,
            softmax_scale: float = 1.0,
            mask_mode: int = 1,
            softmax_precision: int = 0,
            win_left: int = -1,
            win_right: int = -1,
            residual_block_mode: int = 0,
            is_consistent_topk: bool = False,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            if layout_sparse_pattern == 0:
                _check_qsag(
                    q,
                    k,
                    v,
                    dout,
                    attn_out,
                    softmax_lse,
                    sparse_block_idx,
                    sparse_block_count,
                    block_shape,
                    metadata=metadata,
                    attn_mask=attn_mask,
                    cu_q=cu_seqlens_q,
                    cu_kv=cu_seqlens_kv,
                    used_q=seqused_q,
                    used_kv=seqused_kv,
                    layout_q=layout_q,
                    layout_kv=layout_kv,
                    residual_block_mode=residual_block_mode,
                    is_consistent_topk=is_consistent_topk,
                    scale=softmax_scale,
                    mask=_resolve_mask_mode(mask_mode),
                    precision=softmax_precision,
                    left=win_left,
                    right=win_right,
                )
            else:
                _check_sparse_attributes(layout_sparse_pattern, residual_block_mode)
            if len(block_shape) != 2:
                raise ValueError(
                    f"{_OP_PREFIX}: block_shape is required and must be [block_x, block_y]"
                )
            del (
                is_consistent_topk,
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
    layout_q: str = "TND",
    layout_kv: str = "TND",
    layout_sparse_pattern: int = 1,
    mask_mode: Union[int, MaskMode, str] = MaskMode.CAUSAL,
    softmax_precision: int = 0,
    win_left: int = -1,
    win_right: int = -1,
    residual_block_mode: Union[
        int, ResidualBlockMode, str
    ] = ResidualBlockMode.MARKED_BY_SPARSE_BLK_IDX,
    is_consistent_topk: bool = False,
) -> torch.Tensor:
    """生成 GBSAG Metadata；residual_block_mode 支持模式 0 的枚举、整数或枚举名字符串。"""
    residual_block_mode = _resolve_residual_block_mode(residual_block_mode)
    _check_sparse_attributes(layout_sparse_pattern, residual_block_mode)
    if len(block_shape) != 2:
        raise ValueError(
            f"{_OP_PREFIX}: block_shape is required and must be [block_x, block_y]"
        )
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
        layout_q,
        layout_kv,
        layout_sparse_pattern,
        mask_mode_i,
        softmax_precision,
        win_left,
        win_right,
        int(residual_block_mode),
        is_consistent_topk,
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
    layout_q: str = "TND",
    layout_kv: str = "TND",
    layout_sparse_pattern: int = 1,
    softmax_scale: float = 1.0,
    mask_mode: Union[int, MaskMode, str] = MaskMode.CAUSAL,
    softmax_precision: int = 0,
    win_left: int = -1,
    win_right: int = -1,
    residual_block_mode: Union[
        int, ResidualBlockMode, str
    ] = ResidualBlockMode.MARKED_BY_SPARSE_BLK_IDX,
    is_consistent_topk: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """计算 GBSAG/QSAG 梯度；QSAG 使用 layout=0、residual=1，支持两种 TopK 提示。"""
    residual_block_mode = _resolve_residual_block_mode(residual_block_mode)
    if layout_sparse_pattern == 0:
        _check_qsag(
            q,
            k,
            v,
            dout,
            attn_out,
            softmax_lse,
            sparse_block_idx,
            sparse_block_count,
            block_shape,
            metadata=metadata,
            attn_mask=attn_mask,
            cu_q=cu_seqlens_q,
            cu_kv=cu_seqlens_kv,
            used_q=seqused_q,
            used_kv=seqused_kv,
            layout_q=layout_q,
            layout_kv=layout_kv,
            residual_block_mode=residual_block_mode,
            is_consistent_topk=is_consistent_topk,
            scale=softmax_scale,
            mask=_resolve_mask_mode(mask_mode),
            precision=softmax_precision,
            left=win_left,
            right=win_right,
        )
    else:
        _check_sparse_attributes(layout_sparse_pattern, residual_block_mode)
    if len(block_shape) != 2:
        raise ValueError(
            f"{_OP_PREFIX}: block_shape is required and must be [block_x, block_y]"
        )
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
        layout_q,
        layout_kv,
        layout_sparse_pattern,
        softmax_scale,
        mask_mode_i,
        softmax_precision,
        win_left,
        win_right,
        int(residual_block_mode),
        is_consistent_topk,
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
