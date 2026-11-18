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
from .msa_index_score_v2 import (
    _packed_query_meta,
    _resolve_pool_block_size,
    _batch_key128_max_blocks,
    PA_BLOCK_SIZE,
    BLOCK16_SIZE,
)


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
BATCH_LAYOUT_KEYS = ("BSND", "BNSD")
BLOCK_SIZE = 128


def _round_up(value: int, align: int) -> int:
    return (value + align - 1) // align * align


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


class MsaIndexScoreOpBuilder(OpBuilder):
    def __init__(self):
        super(MsaIndexScoreOpBuilder, self).__init__(
            "msa_index_score", category="attention"
        )

    def sources(self):
        """Path to C++ source code."""
        return ["csrc/attention/msa_index_score.cpp"]

    def schema(self) -> str:
        # Positional order must match the public Python API: query, key, start_loc.
        # cann_ops_transformer.__init__ re-exports torch.ops once the schema exists.
        return (
            "msa_index_score(Tensor query, Tensor key, Tensor start_loc, "
            "Tensor? block_table=None, Tensor? query_scale=None, Tensor? key_scale=None, "
            "Tensor? atten_mask=None, Tensor? actual_seq_qlen=None, Tensor? actual_seq_klen=None, "
            '*, str layout_key="BBND", int sparse_mode=3, int init_blocks=0, int local_blocks=1, '
            "int compress_factor=1) -> Tensor"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def msa_index_score_meta(
            query: torch.Tensor,
            key: torch.Tensor,
            start_loc: torch.Tensor,
            block_table: Optional[torch.Tensor] = None,
            query_scale: Optional[torch.Tensor] = None,
            key_scale: Optional[torch.Tensor] = None,
            atten_mask: Optional[torch.Tensor] = None,
            actual_seq_qlen: Optional[torch.Tensor] = None,
            actual_seq_klen: Optional[torch.Tensor] = None,
            *,
            layout_key: str = DEFAULT_LAYOUT_KEY,
            sparse_mode: int = DEFAULT_SPARSE_MODE,
            init_blocks: int = DEFAULT_INIT_BLOCKS,
            local_blocks: int = DEFAULT_LOCAL_BLOCKS,
            compress_factor: int = 1,
        ):
            torch._check(
                compress_factor == 1, lambda: "compress_factor only supports 1"
            )
            layout_query = "TND"
            pool_block_size = 128
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
                    layout == layout_query
                    or (
                        layout in ("BSND", "BNSD")
                        and layout_query == "TND"
                        and pool_block_size == PA_BLOCK_SIZE
                    ),
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
                blk = _resolve_pool_block_size(
                    layout, key, pool_block_size, layout_query
                )
                page = int(key.size(2) if layout == "BNBD" else key.size(1))
                max_blocks = int(block_table.size(1)) * (page // blk)
            else:
                torch._check(
                    actual_seq_klen is not None and actual_seq_klen.dim() == 1,
                    lambda: "layout_key=TND requires actual_seq_klen [B+1]",
                )
                if (
                    layout in ("BSND", "BNSD")
                    and layout_query == "TND"
                    and pool_block_size == PA_BLOCK_SIZE
                ):
                    max_blocks = _batch_key128_max_blocks(
                        query, key, layout, actual_seq_klen, pool_block_size
                    )
                else:
                    pref = [int(v) for v in actual_seq_klen.tolist()]
                    blk = _resolve_pool_block_size(
                        layout, key, pool_block_size, layout_query
                    )
                    max_blocks = 0
                    for i in range(len(pref) - 1):
                        kv = pref[i + 1] - pref[i]
                        blocks = 0 if kv <= 0 else (kv + blk - 1) // blk
                        if blocks > max_blocks:
                            max_blocks = blocks
                    if max_blocks <= 0:
                        max_blocks = 1
            score_stride = _round_up(int(max_blocks), SCORE_STRIDE_ALIGN)
            blk = _resolve_pool_block_size(layout, key, pool_block_size, layout_query)
            score_heads = 1 if blk == BLOCK16_SIZE else num_q_heads
            return torch.empty(
                (score_heads, total_q, score_stride),
                dtype=torch.float32,
                device="meta",
            )


_msa_index_score_op_builder = MsaIndexScoreOpBuilder()
_msa_index_score_op_builder._ensure_initialized()
_op_module = None


def _get_op_module():
    global _op_module
    if _op_module is None:
        _op_module = _msa_index_score_op_builder.load()
    return _op_module


@impl(get_as_library(), _msa_index_score_op_builder.name, "PrivateUse1")
def _msa_index_score(
    query: torch.Tensor,
    key: torch.Tensor,
    start_loc: torch.Tensor,
    block_table: Optional[torch.Tensor] = None,
    query_scale: Optional[torch.Tensor] = None,
    key_scale: Optional[torch.Tensor] = None,
    atten_mask: Optional[torch.Tensor] = None,
    actual_seq_qlen: Optional[torch.Tensor] = None,
    actual_seq_klen: Optional[torch.Tensor] = None,
    *,
    layout_key: str = DEFAULT_LAYOUT_KEY,
    sparse_mode: int = DEFAULT_SPARSE_MODE,
    init_blocks: int = DEFAULT_INIT_BLOCKS,
    local_blocks: int = DEFAULT_LOCAL_BLOCKS,
    compress_factor: int = 1,
) -> torch.Tensor:
    layout = _normalize_layout_key(layout_key)
    if layout == "BBND" and key.dim() == 3 and int(key.size(1)) == 128:
        key = key.unsqueeze(2)
    return _get_op_module().msa_index_score(
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
        compress_factor,
    )


def msa_index_score(
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
    return_choice: int = 0,
    topk: int = 16,
    compress_factor: int = 1,
):
    """MSA Index Branch block score，封装 aclnnMsaIndexScore。

    950 额外支持 query/key 同型
    ``torch_npu.hifloat8`` / ``torch.float8_e5m2`` / ``torch.float8_e4m3fn``（无 scale）。
    key 布局由 ``layout_key`` 指定：PA BBND ``[NP, P, N2, D]``、BNBD ``[NP, N2, P, D]``、
    TND ``[T2, N2, D]``、BSND ``[B, S2, N2, D]``、BNSD ``[B, N2, S2, D]``。
    query 固定 TND。BSND / BNSD 不传 ``block_table``，``actual_seq_klen`` 为 ``[B+1]`` 前缀和，逻辑块长 128。
    允许 ``q_len`` / ``kv_len`` 为 0（含整 batch）：跳过该请求的 QK；Ascend 950 上空 KV 的 score 为 ``-FLT_MAX``。

    Args:
        query: ``[T1, N1, D]`` float16 / bfloat16；950 还可为三种 FP8
        key: 与 ``layout_key`` 对应；非量化须与 query 同 dtype；int8 仅 fp16 query + key_scale
        start_loc: ``[B]``，当前 query 所在逻辑 block 索引（local_mask）
        block_table: ``[B, MB]``，BBND/BNBD 必选；TND / BSND / BNSD 不传
        key_scale: MXFP4 为独立 K E8M0 group32 系数；int8 key 反量化时 PA 为 ``[NP, N2, P]``；TND 为 ``[T2, N2]``；非量化 / FP8 为 None
        query_scale: MXFP4 必填的独立 Q E8M0 group32 系数；其他类型为 None
        compress_factor: key 压缩倍数，1 表示不压缩，当前只支持 1
        atten_mask: sparse_mode=3 时 ``[2048, 2048]`` int8
        actual_seq_qlen: query 前缀和 ``[B+1]``
        actual_seq_klen: PA 为各请求可见长度 ``[B]``；TND / BSND / BNSD 为 key 前缀和 ``[B+1]``
        layout_key: ``TND`` / ``BSND`` / ``BNSD`` / ``BBND`` / ``BNBD``，默认 ``BBND``
        sparse_mode: 0 或 3
        init_blocks / local_blocks: local_mask 强制选块；与 Triton raw score 对齐时置 0

    Returns:
        ``return_choice=0``：``[N1, T1, RoundUp(MB, 16)]`` float32。
        ``return_choice=1``：``[N1, T1, topk]`` int32。
        ``return_choice=2``：indices 与同 shape 的 float32 values。
    """
    if return_choice != 0:
        outputs = _get_op_module().msa_index_score_sparse(
            query,
            key,
            start_loc,
            block_table,
            query_scale,
            key_scale,
            atten_mask,
            actual_seq_qlen,
            actual_seq_klen,
            _normalize_layout_key(layout_key),
            sparse_mode,
            init_blocks,
            local_blocks,
            return_choice,
            topk,
            compress_factor,
        )
        if return_choice == 1:
            return outputs[0]
        return outputs[0], outputs[1]
    return torch.ops.cann_ops_transformer.msa_index_score(
        query,
        key,
        start_loc,
        block_table,
        query_scale,
        key_scale,
        atten_mask,
        actual_seq_qlen,
        actual_seq_klen,
        layout_key=_normalize_layout_key(layout_key),
        sparse_mode=sparse_mode,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        compress_factor=compress_factor,
    )
