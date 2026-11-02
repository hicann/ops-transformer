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

SCORE_STRIDE_ALIGN = 16
DEFAULT_SPARSE_MODE = 3
DEFAULT_INIT_BLOCKS = 0
DEFAULT_LOCAL_BLOCKS = 1
DEFAULT_LAYOUT_KEY = "BBND"
SUPPORTED_LAYOUT_KEYS = ("TND", "BSND", "BNSD", "BBND", "BNBD")
SEQ_LAYOUT_KEYS = ("TND", "BSND", "BNSD")
# BSND/BNSD，以及 BBND/BNBD 页长 16。TND 固定 128，每个 head 只做一次块内 maxpool。
BLOCK16_SIZE = 16
PA_BLOCK_SIZE = 128


def _resolve_pool_block_size(
    layout: str, key: torch.Tensor, pool_block_size: int
) -> int:
    """pool_block_size 只允许 16 / 128，不做推导；BSND/BNSD 固定 16。
    存储页长（key 的页维度）与打分块长解耦，但打分块不能超过存储页容量。"""
    pool_block_size = int(pool_block_size)
    if pool_block_size not in (BLOCK16_SIZE, PA_BLOCK_SIZE):
        raise ValueError(
            f"pool_block_size must be {BLOCK16_SIZE} or {PA_BLOCK_SIZE}, got {pool_block_size}"
        )
    if layout in ("BSND", "BNSD"):
        if pool_block_size != BLOCK16_SIZE:
            raise ValueError(
                f"BSND/BNSD requires pool_block_size={BLOCK16_SIZE}, got {pool_block_size}"
            )
        return pool_block_size
    if layout not in SEQ_LAYOUT_KEYS:
        page = int(key.size(2) if layout == "BNBD" else key.size(1))
        if pool_block_size > page:
            raise ValueError(
                f"pool_block_size({pool_block_size}) must not exceed BBND/BNBD page size({page})"
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
            "msa_index_score_v2", category="attention"
        )

    def sources(self):
        """Path to C++ source code."""
        return ["csrc/attention/msa_index_score_v2.cpp"]

    def schema(self) -> str:
        # Positional order must match the public Python API: query, key, start_loc.
        # cann_ops_transformer.__init__ re-exports torch.ops once the schema exists.
        return (
            "msa_index_score_v2(Tensor query, Tensor key, Tensor start_loc, "
            "Tensor? block_table=None, Tensor? scale=None, Tensor? atten_mask=None, "
            'Tensor? actual_seq_qlen=None, Tensor? actual_seq_klen=None, *, str layout_key="BBND", '
            'int sparse_mode=3, int init_blocks=0, int local_blocks=1, str layout_query="TND", '
            "int pool_block_size=128) -> Tensor"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def msa_index_score_meta(
            query: torch.Tensor,
            key: torch.Tensor,
            start_loc: torch.Tensor,
            block_table: Optional[torch.Tensor],
            scale: Optional[torch.Tensor],
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
            if layout in SEQ_LAYOUT_KEYS:
                torch._check(
                    layout == layout_query,
                    lambda: "non-paged key layout must match layout_query",
                )
            if layout not in SEQ_LAYOUT_KEYS:
                torch._check(
                    block_table is not None,
                    lambda: f"layout_key={layout} requires block_table",
                )
                torch._check(
                    block_table.dim() == 2, lambda: "block_table must be 2D [B, MB]"
                )
                max_blocks = int(block_table.size(1))
            else:
                torch._check(
                    actual_seq_klen is not None and actual_seq_klen.dim() == 1,
                    lambda: "layout_key=TND requires actual_seq_klen [B+1]",
                )
                pref = [int(v) for v in actual_seq_klen.tolist()]
                blk = _resolve_pool_block_size(layout, key, pool_block_size)
                max_blocks = 0
                for i in range(len(pref) - 1):
                    kv = pref[i + 1] - pref[i]
                    blocks = 0 if kv <= 0 else (kv + blk - 1) // blk
                    if blocks > max_blocks:
                        max_blocks = blocks
                if max_blocks <= 0:
                    max_blocks = 1
            score_stride = _round_up(int(max_blocks), SCORE_STRIDE_ALIGN)
            blk = _resolve_pool_block_size(layout, key, pool_block_size)
            score_heads = 1 if blk == BLOCK16_SIZE else num_q_heads
            return torch.empty(
                (score_heads, total_q, score_stride),
                dtype=torch.float32,
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
    scale: Optional[torch.Tensor],
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
) -> torch.Tensor:
    layout = _normalize_layout_key(layout_key)
    layout_query = _normalize_layout_query(layout_query)
    pool_block_size = _resolve_pool_block_size(layout, key, pool_block_size)
    if layout == "BBND" and key.dim() == 3 and int(key.size(1)) == 128:
        key = key.unsqueeze(2)
    # C++ wrapper still follows aclnn order: optionals then start_loc.
    return _get_op_module().msa_index_score_v2(
        query,
        key,
        block_table,
        scale,
        atten_mask,
        actual_seq_qlen,
        actual_seq_klen,
        start_loc,
        layout,
        sparse_mode,
        init_blocks,
        local_blocks,
        layout_query,
        pool_block_size,
    )


def msa_index_score_v2(
    query: torch.Tensor,
    key: torch.Tensor,
    start_loc: torch.Tensor,
    *,
    block_table: Optional[torch.Tensor] = None,
    scale: Optional[torch.Tensor] = None,
    atten_mask: Optional[torch.Tensor] = None,
    actual_seq_qlen: Optional[torch.Tensor] = None,
    actual_seq_klen: Optional[torch.Tensor] = None,
    layout_key: str = DEFAULT_LAYOUT_KEY,
    sparse_mode: int = DEFAULT_SPARSE_MODE,
    init_blocks: int = DEFAULT_INIT_BLOCKS,
    local_blocks: int = DEFAULT_LOCAL_BLOCKS,
    layout_query: str = "TND",
    pool_block_size: int = 128,
) -> torch.Tensor:
    """MSA Index Branch block score，封装 aclnnMsaIndexScoreV2。

    950 额外支持 query/key 同型
    ``torch_npu.hifloat8`` / ``torch.float8_e5m2`` / ``torch.float8_e4m3fn``（无 scale）。
    key 布局由 ``layout_key`` 指定：PA BBND ``[NP, P, N2, D]``、BNBD ``[NP, N2, P, D]``、
    TND ``[T2, N2, D]``、BSND ``[B, S2, N2, D]``、BNSD ``[B, N2, S2, D]``。
    query 由 ``layout_query`` 指定 TND / BSND / BNSD，默认 TND。非分页 key 必须与 query 相同；
    分页 key 可配任意 query。BSND、BNSD 仅 Ascend 950，逻辑块长 16，仅 fp16 / bf16。
    ``pool_block_size`` 是每个 score 归约的 token 数，只允许 16 / 128，默认 128。
    与 BBND / BNBD 的存储页长不是同一个量；分页时不能超过页长。
    128 走原单次块内 maxpool（TND 默认）；16 仅 Ascend 950、fp16 / bf16，
    先在每个 head 内对 block 取 max，再对每个 block 取 head 间 max，
    score 的 head 维为 1。query 的 head 数必须整除 128。
    允许 ``q_len`` / ``kv_len`` 为 0（含整 batch）：跳过该请求的 QK；空 KV 的 score 为 ``-inf``。

    Args:
        query: ``[T1, N1, D]`` float16 / bfloat16；950 还可为三种 FP8
        key: 与 ``layout_key`` 对应；非量化须与 query 同 dtype；int8 仅 fp16 query + scale
        start_loc: ``[B]``，当前 query 所在逻辑 block 索引（local_mask）
        block_table: ``[B, MB]``，BBND/BNBD 必选；TND 不传
        scale: int8 反量化。PA 为 ``[NP, N2, P]``；TND 为 ``[T2, N2]``；非量化 / FP8 为 None
        atten_mask: sparse_mode=3 时 ``[2048, 2048]`` int8
        actual_seq_qlen: query 前缀和 ``[B+1]``
        actual_seq_klen: PA 为各请求 S2 ``[B]``；TND 为 key 前缀和 ``[B+1]``
        layout_key: ``TND`` / ``BSND`` / ``BNSD`` / ``BBND`` / ``BNBD``，默认 ``BBND``
        sparse_mode: 0 或 3
        init_blocks / local_blocks: local_mask 强制选块；与 Triton raw score 对齐时置 0
        pool_block_size: 打分池化块长，只允许 16 / 128，默认 128。BSND / BNSD 必须为 16；
            BBND / BNBD 不能超过存储页长。

    Returns:
        ``pool_block_size=16`` 时 ``[1, T1, RoundUp(MB, 16)]``，否则 ``[N1, T1, RoundUp(MB, 16)]``，float32
    """
    layout_query = _normalize_layout_query(layout_query)
    layout_key = _normalize_layout_key(layout_key)
    if layout_key in SEQ_LAYOUT_KEYS and layout_key != layout_query:
        raise ValueError(
            f"non-paged key layout must match layout_query, got query={layout_query} key={layout_key}"
        )
    pool_block_size = _resolve_pool_block_size(layout_key, key, pool_block_size)
    return torch.ops.cann_ops_transformer.msa_index_score_v2(
        query,
        key,
        start_loc,
        block_table,
        scale,
        atten_mask,
        actual_seq_qlen,
        actual_seq_klen,
        layout_key=layout_key,
        sparse_mode=sparse_mode,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        layout_query=layout_query,
        pool_block_size=pool_block_size,
    )
