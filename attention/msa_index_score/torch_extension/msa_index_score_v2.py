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
from torch._subclasses.fake_tensor import is_fake
from torch.library import impl
from cann_ops_transformer.op_builder import OpBuilder, get_as_library

SCORE_STRIDE_ALIGN = 16
DEFAULT_SPARSE_MODE = 3
DEFAULT_INIT_BLOCKS = 0
DEFAULT_LOCAL_BLOCKS = 1
DEFAULT_LAYOUT_KEY = "BBND"
SUPPORTED_LAYOUT_KEYS = ("TND", "BSND", "BNSD", "BBND", "BNBD")
SEQ_LAYOUT_KEYS = ("TND", "BSND", "BNSD")
# BSND/BNSD，以及 BBND/BNBD 页长 16。TND 固定 128，每个 head 只做一次块内 maxpool。
BLOCK16_SIZE = 16
BLOCK16_TOPK = 128
PA_BLOCK_SIZE = 128


def _batch_key128_max_blocks(query, key, layout, actual_seq_klen, pool_block_size):
    seq_cap = key.size(1) if layout == "BSND" else key.size(2)
    capacity_blocks = max(1, (seq_cap + pool_block_size - 1) // pool_block_size)
    mx_dtype = getattr(torch, "float4_e2m1fn_x2", None)
    fake_lengths = is_fake(actual_seq_klen)
    lengths_device = (
        actual_seq_klen.fake_device if fake_lengths else actual_seq_klen.device
    )
    if query.dtype == mx_dtype or lengths_device.type != "cpu":
        return capacity_blocks
    if fake_lengths:
        raise RuntimeError(
            "native BSND/BNSD block128 output shape depends on CPU actual_seq_klen values; "
            "these values are unavailable for FakeTensor graph tracing"
        )
    pref = actual_seq_klen.tolist()
    max_blocks = 0
    for left, right in zip(pref, pref[1:]):
        kv = int(right) - int(left)
        torch._check(0 <= kv <= seq_cap, lambda: "actual_seq_klen exceeds key capacity")
        max_blocks = max(max_blocks, (kv + pool_block_size - 1) // pool_block_size)
    return max(1, max_blocks)


def _is_batch_key128(layout, query_layout, pool_block_size):
    return (
        layout in ("BSND", "BNSD")
        and query_layout == "TND"
        and int(pool_block_size) == PA_BLOCK_SIZE
    )


def _resolve_pool_block_size(
    layout: str,
    key: torch.Tensor,
    pool_block_size: int,
    query_layout: str = "TND",
    *,
    compress_factor: int = 1,
) -> int:
    pool_block_size = int(pool_block_size)
    if pool_block_size not in (BLOCK16_SIZE, PA_BLOCK_SIZE):
        raise ValueError(f"pool_block_size must be 16 or 128, got {pool_block_size}")
    cf = int(compress_factor)
    if cf < 1 or pool_block_size % cf or (pool_block_size == PA_BLOCK_SIZE and cf != 1):
        raise ValueError(
            "compress_factor must divide pool_block_size and compression requires pool_block_size=16"
        )
    if layout in ("BSND", "BNSD"):
        if pool_block_size != BLOCK16_SIZE and not _is_batch_key128(
            layout, query_layout, pool_block_size
        ):
            raise ValueError("pool_block_size=128 requires TND query for BSND/BNSD key")
    elif layout not in SEQ_LAYOUT_KEYS:
        page = int(key.size(2) if layout == "BNBD" else key.size(1))
        rows_per_block = pool_block_size // cf
        if page < rows_per_block or page % rows_per_block:
            raise ValueError(
                f"pool_block_size/compress_factor({rows_per_block}) must divide physical page rows({page})"
            )
    return pool_block_size


def _round_up(value: int, align: int) -> int:
    return (value + align - 1) // align * align


def _normalize_layout_query(layout_query: Optional[str]) -> str:
    layout = (
        "TND"
        if layout_query is None or layout_query == ""
        else str(layout_query).upper()
    )
    if layout not in ("TND", "BSND", "BNSD"):
        raise ValueError(f"layout_query must be TND, BSND or BNSD, got {layout_query}")
    return layout


def _packed_query_meta(
    query: torch.Tensor, layout_query: str, actual_seq_qlen: Optional[torch.Tensor]
):
    """score 的 T1 是 actual_seq_qlen 的有效 token 数，不是 BSND/BNSD 的 B*S。"""
    if layout_query == "BSND":
        num_q_heads = int(query.size(2))
        seq_cap = int(query.size(1))
    elif layout_query == "BNSD":
        num_q_heads = int(query.size(1))
        seq_cap = int(query.size(2))
    else:
        return int(query.size(0)), int(query.size(1))
    torch._check(
        actual_seq_qlen is not None, lambda: "BSND/BNSD requires actual_seq_qlen"
    )
    pref = [int(v) for v in actual_seq_qlen.tolist()]
    for i in range(len(pref) - 1):
        qlen = pref[i + 1] - pref[i]
        torch._check(
            0 <= qlen <= seq_cap,
            lambda qlen=qlen,
            seq_cap=seq_cap: f"query length {qlen} exceeds layout S={seq_cap}",
        )
    return int(pref[-1]) if pref else 0, num_q_heads


def _normalize_layout_key(layout_key: Optional[str]) -> str:
    layout = (
        DEFAULT_LAYOUT_KEY
        if layout_key is None or layout_key == ""
        else str(layout_key).upper()
    )
    if layout not in SUPPORTED_LAYOUT_KEYS:
        raise ValueError(
            f"layout_key must be TND, BSND, BNSD, BBND or BNBD, got {layout_key}"
        )
    return layout


class MsaIndexScoreV2OpBuilder(OpBuilder):
    def __init__(self):
        super(MsaIndexScoreV2OpBuilder, self).__init__(
            "_msa_index_score_v2", category="attention"
        )

    def sources(self):
        """Path to C++ source code."""
        return ["csrc/attention/msa_index_score_v2.cpp"]

    def schema(self) -> str:
        # Keep the Python entry point separate from the single-output dispatcher.
        return (
            "_msa_index_score_v2(Tensor query, Tensor key, Tensor start_loc, "
            "Tensor? block_table=None, Tensor? query_scale=None, Tensor? key_scale=None, Tensor? atten_mask=None, "
            "Tensor? actual_seq_qlen=None, Tensor? actual_seq_klen=None, "
            '*, str layout_key="BBND", '
            'int sparse_mode=3, int init_blocks=0, int local_blocks=1, str layout_query="TND", '
            "int pool_block_size=128, int return_choice=0, int topk=16, int compress_factor=1) -> Tensor"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def msa_index_score_meta(
            query: torch.Tensor,
            key: torch.Tensor,
            start_loc: torch.Tensor,
            block_table: Optional[torch.Tensor],
            query_scale: Optional[torch.Tensor],
            key_scale: Optional[torch.Tensor],
            atten_mask: Optional[torch.Tensor],
            actual_seq_qlen: Optional[torch.Tensor],
            actual_seq_klen: Optional[torch.Tensor],
            *,
            layout_key: str = DEFAULT_LAYOUT_KEY,
            sparse_mode: int = DEFAULT_SPARSE_MODE,
            init_blocks: int = DEFAULT_INIT_BLOCKS,
            local_blocks: int = DEFAULT_LOCAL_BLOCKS,
            layout_query: str = "TND",
            pool_block_size: int = 128,
            return_choice: int = 0,
            topk: int = 16,
            compress_factor: int = 1,
        ):
            layout_query = _normalize_layout_query(layout_query)
            torch._check(
                query.dim() == (3 if layout_query == "TND" else 4),
                lambda: "query rank does not match layout_query",
            )
            total_q, num_q_heads = _packed_query_meta(
                query, layout_query, actual_seq_qlen
            )
            layout = _normalize_layout_key(layout_key)
            batch_key128 = _is_batch_key128(layout, layout_query, pool_block_size)
            if layout in SEQ_LAYOUT_KEYS and not batch_key128:
                torch._check(
                    layout == layout_query,
                    lambda: "non-paged key layout must match layout_query",
                )
            blk = _resolve_pool_block_size(
                layout,
                key,
                pool_block_size,
                layout_query,
                compress_factor=compress_factor,
            )
            if int(return_choice) in (1, 2):
                torch._check(
                    1 <= int(topk) <= 128
                    and (blk != BLOCK16_SIZE or int(topk) == BLOCK16_TOPK),
                    lambda: "Invalid topk for the selected pool_block_size",
                )
                return torch.empty(
                    (1 if blk == BLOCK16_SIZE else num_q_heads, total_q, int(topk)),
                    dtype=torch.int32 if int(return_choice) == 1 else torch.float32,
                    device="meta",
                )
            if layout not in SEQ_LAYOUT_KEYS:
                torch._check(
                    block_table is not None,
                    lambda: f"layout_key={layout} requires block_table",
                )
                torch._check(
                    block_table.dim() == 2, lambda: "block_table must be 2D [B, MB]"
                )
                page = int(key.size(2) if layout == "BNBD" else key.size(1))
                max_blocks = max(
                    1,
                    int(block_table.size(1)) * (page // (blk // int(compress_factor))),
                )
            else:
                torch._check(
                    actual_seq_klen is not None and actual_seq_klen.dim() == 1,
                    lambda: "non-paged key requires actual_seq_klen [B+1]",
                )
                if batch_key128:
                    max_blocks = _batch_key128_max_blocks(
                        query, key, layout, actual_seq_klen, blk
                    )
                else:
                    pref = [int(v) for v in actual_seq_klen.tolist()]
                    max_blocks = max(
                        1,
                        max(
                            ((r - l + blk - 1) // blk for l, r in zip(pref, pref[1:])),
                            default=0,
                        ),
                    )
            score_stride = _round_up(int(max_blocks), SCORE_STRIDE_ALIGN)
            score_heads = 1 if blk == BLOCK16_SIZE else num_q_heads
            score_dtype = torch.float16 if blk == BLOCK16_SIZE else torch.float32
            return torch.empty(
                (score_heads, total_q, score_stride),
                dtype=score_dtype,
                device="meta",
            )


_msa_index_score_v2_op_builder = MsaIndexScoreV2OpBuilder()
_msa_index_score_v2_op_builder._ensure_initialized()
_op_module = None


def _get_op_module():
    global _op_module
    if _op_module is None:
        _op_module = _msa_index_score_v2_op_builder.load()
    return _op_module


@impl(get_as_library(), _msa_index_score_v2_op_builder.name, "PrivateUse1")
def _msa_index_score_v2(
    query: torch.Tensor,
    key: torch.Tensor,
    start_loc: torch.Tensor,
    block_table: Optional[torch.Tensor],
    query_scale: Optional[torch.Tensor],
    key_scale: Optional[torch.Tensor],
    atten_mask: Optional[torch.Tensor],
    actual_seq_qlen: Optional[torch.Tensor],
    actual_seq_klen: Optional[torch.Tensor],
    *,
    layout_key: str = DEFAULT_LAYOUT_KEY,
    sparse_mode: int = DEFAULT_SPARSE_MODE,
    init_blocks: int = DEFAULT_INIT_BLOCKS,
    local_blocks: int = DEFAULT_LOCAL_BLOCKS,
    layout_query: str = "TND",
    pool_block_size: int = 128,
    return_choice: int = 0,
    topk: int = 16,
    compress_factor: int = 1,
) -> torch.Tensor:
    layout = _normalize_layout_key(layout_key)
    layout_query = _normalize_layout_query(layout_query)
    pool_block_size = _resolve_pool_block_size(
        layout,
        key,
        pool_block_size,
        layout_query,
        compress_factor=compress_factor,
    )
    if layout == "BBND" and key.dim() == 3 and int(key.size(1)) == 128:
        key = key.unsqueeze(2)
    # The C++ wrapper follows the PR15456 tensor order.
    score, indices, _values = _get_op_module().msa_index_score_v2(
        query,
        key,
        start_loc,
        block_table,
        query_scale,
        key_scale,
        atten_mask,
        actual_seq_qlen,
        actual_seq_klen,
        layout,
        sparse_mode,
        init_blocks,
        local_blocks,
        layout_query,
        pool_block_size,
        return_choice,
        topk,
        compress_factor,
    )
    return indices if int(return_choice) != 0 else score


def msa_index_score_v2(
    query: torch.Tensor,
    key: torch.Tensor,
    start_loc: torch.Tensor,
    *,
    block_table: Optional[torch.Tensor] = None,
    query_scale: Optional[torch.Tensor] = None,
    key_scale: Optional[torch.Tensor] = None,
    atten_mask: Optional[torch.Tensor] = None,
    actual_seq_qlen: Optional[torch.Tensor] = None,
    actual_seq_klen: Optional[torch.Tensor] = None,
    layout_key: str = DEFAULT_LAYOUT_KEY,
    sparse_mode: int = DEFAULT_SPARSE_MODE,
    init_blocks: int = DEFAULT_INIT_BLOCKS,
    local_blocks: int = DEFAULT_LOCAL_BLOCKS,
    layout_query: str = "TND",
    pool_block_size: int = 128,
    return_choice: int = 0,
    topk: int = 16,
    compress_factor: int = 1,
):
    """调用 aclnnMsaIndexScoreV2，计算每个逻辑块的分数或返回 TopK。

    query 支持 TND、BSND、BNSD。非分页 key 与 query 布局一致，
    块长 128 也支持 TND query 配 BSND/BNSD key。分页 key 支持 BBND、BNBD，
    需要传入 block_table。长度张量可放在 CPU 或 NPU，图捕获时使用 NPU 长度张量。

    块长 128 按 head 取块内最大值。Ascend 950 支持块长 16，先取块内最大值，
    再取跨 head 最大值并缩放。FP8 的 query 与 key 类型相同且不传 scale，
    MXFP4 则需要分别传入 query_scale 和 key_scale（每 32 个元素一组 E8M0）。
    块长 16 支持 compress_factor > 1，实际序列长度仍按压缩前的 token 数传入。

    return_choice 为 0 时返回 score，块长 128 为 float32，块长 16 为 float16。
    在 Ascend 950 上，为 1 时返回 int32 索引，为 2 时返回索引和 float32 分值。
    块长 128 的 topk 支持 1 到 128，块长 16 的 topk 固定为 128。
    """
    layout_query = _normalize_layout_query(layout_query)
    layout_key = _normalize_layout_key(layout_key)
    if (
        layout_key in SEQ_LAYOUT_KEYS
        and layout_key != layout_query
        and not _is_batch_key128(layout_key, layout_query, pool_block_size)
    ):
        raise ValueError(
            f"non-paged key layout must match layout_query, got query={layout_query} key={layout_key}"
        )
    pool_block_size = _resolve_pool_block_size(
        layout_key,
        key,
        pool_block_size,
        layout_query,
        compress_factor=compress_factor,
    )
    if int(return_choice) == 2:
        _score, indices, values = _get_op_module().msa_index_score_v2(
            query,
            key,
            start_loc,
            block_table,
            query_scale,
            key_scale,
            atten_mask,
            actual_seq_qlen,
            actual_seq_klen,
            layout_key,
            sparse_mode,
            init_blocks,
            local_blocks,
            layout_query,
            pool_block_size,
            int(return_choice),
            int(topk),
            int(compress_factor),
        )
        return indices, values
    return torch.ops.cann_ops_transformer._msa_index_score_v2(
        query,
        key,
        start_loc,
        block_table,
        query_scale,
        key_scale,
        atten_mask,
        actual_seq_qlen,
        actual_seq_klen,
        layout_key=layout_key,
        sparse_mode=sparse_mode,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        layout_query=layout_query,
        pool_block_size=pool_block_size,
        return_choice=int(return_choice),
        topk=int(topk),
        compress_factor=int(compress_factor),
    )
