#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""MsaIndexScoreV2 CPU 参考实现。分页可见块停在 block_table 宽度，超出的列保持 -inf。

数值路径（与 kernel 对齐）：
  score = Maxpool[ (scale ·) Q @ Kᵀ + atten_mask ] + local_mask

- sparse_mode=0：无因果，仅 actual_seq_klen 截断
- sparse_mode=3：rightDownCausal，visible_end = actual_seq_klen - q_len + t_off + 1
- local_mask：窗口锚在请求的 start_loc 上，同一请求的每个 token 共用。
  另外要求 blk < visibleEndBlk。visibleEndBlk 用该行所在 128 行 M-tile 的最后一行计算。
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

NEG_INF = np.float32(-3.4028234663852886e38)
LOCAL_SCORE_INIT = np.float32(1.0e30)
LOCAL_SCORE_LOCAL = np.float32(1.0e29)
SCORE_STRIDE_ALIGN = 16
SPARSE_MODE_DEFAULT = 0
SPARSE_MODE_RIGHT_DOWN = 3
DEFAULT_INIT_BLOCKS = 0
DEFAULT_LOCAL_BLOCKS = 1
DEFAULT_BLOCK_SIZE = 128


@dataclass
class MsaIndexScoreGoldenInputs:
    """msa_index_score_golden 的具名入参（tensor + attr）。"""

    query: np.ndarray
    key: np.ndarray
    block_table: Optional[np.ndarray]
    actual_seq_qlen: np.ndarray
    actual_seq_klen: np.ndarray
    start_loc: np.ndarray
    sparse_mode: int = SPARSE_MODE_RIGHT_DOWN
    pool_block_size: int = DEFAULT_BLOCK_SIZE
    scale: Optional[np.ndarray] = None
    init_blocks: int = DEFAULT_INIT_BLOCKS
    local_blocks: int = DEFAULT_LOCAL_BLOCKS
    # None 表示按 key 维数推断 TND 或 PA。
    layout: Optional[str] = None


def round_up(value, align):
    return (value + align - 1) // align * align


ROW_TILE_M = 128
M_ALIGN = 16


def _visible_key_end(sparse_mode, actual_seq_klen, q_len, t_off):
    if sparse_mode == SPARSE_MODE_RIGHT_DOWN:
        return int(np.clip(actual_seq_klen - q_len + t_off + 1, 0, actual_seq_klen))
    return int(actual_seq_klen)


def _tile_visible_end_blk(
    row, total_rows, num_heads, sparse_mode, kv, q_len, pool_block_size, num_blocks
):
    """与 kernel VisibleEndBlkOf 一致。尾 tile 不足 16 行时向前重叠，后写的 tile 生效。"""
    if total_rows <= 0 or num_heads <= 0 or pool_block_size <= 0:
        return 0
    end_blk = 0
    n_tiles = (total_rows + ROW_TILE_M - 1) // ROW_TILE_M
    for tile in range(n_tiles):
        m_start = tile * ROW_TILE_M
        m_actual = min(ROW_TILE_M, total_rows - m_start)
        if m_actual <= 0:
            continue
        if m_actual < M_ALIGN and m_start >= M_ALIGN:
            m_start -= M_ALIGN - m_actual
            m_actual = M_ALIGN
        if row < m_start or row >= m_start + m_actual:
            continue
        t_hi = (m_start + m_actual - 1) // num_heads
        visible = _visible_key_end(sparse_mode, kv, q_len, t_hi)
        end_blk = min(
            (max(visible, 0) + pool_block_size - 1) // pool_block_size, num_blocks
        )
    return end_blk


def msa_index_score_golden_v2(inputs):
    """计算 MSA index block score。

    Args:
        inputs: MsaIndexScoreGoldenInputs
            query:       [T1, N1, D]
            key:         PA BBND [NP, P, N2, D]、BNBD [NP, N2, P, D]，或 TND [T2, N2, D]
            block_table: PA [B, MB]；TND 为 None
            actual_seq_qlen:    [B+1]
            actual_seq_klen:    PA [B]；TND 前缀和 [B+1]
            start_loc:   [B]，当前 query 所在逻辑 block 索引（local_mask）
            sparse_mode: 0 或 3
            scale:       int8 时 PA [NP, N2, P] 或 TND [T2, N2]；非量化为 None
    """
    query = inputs.query
    key = inputs.key
    block_table = inputs.block_table
    actual_seq_qlen = inputs.actual_seq_qlen
    actual_seq_klen = inputs.actual_seq_klen
    start_loc = inputs.start_loc
    sparse_mode = inputs.sparse_mode
    pool_block_size = inputs.pool_block_size
    scale = inputs.scale
    init_blocks = inputs.init_blocks
    local_blocks = inputs.local_blocks

    total_q, num_q_heads, head_dim = query.shape
    is_tnd = key.ndim == 3
    if is_tnd:
        batch = len(actual_seq_klen) - 1
        max_blocks = 0
        for b in range(batch):
            kv = int(actual_seq_klen[b + 1]) - int(actual_seq_klen[b])
            blocks = (kv + pool_block_size - 1) // pool_block_size
            if blocks > max_blocks:
                max_blocks = blocks
        num_kv_heads = key.shape[1]
        k = key.astype(np.float32)[:, 0, :]
    else:
        batch = len(actual_seq_klen)
        # N2=1 时 BBND [NP,P,N2,D] 的 dim2=1、BNBD [NP,N2,P,D] 的 dim1=1，按此判别布局。
        is_bnbd = key.shape[1] == 1 and key.shape[2] > 1
        num_kv_heads = key.shape[1] if is_bnbd else key.shape[2]
        if is_bnbd:
            k = key.astype(np.float32)[:, 0, :, :]
        else:
            k = key.astype(np.float32)[:, :, 0, :]
        # 分页逻辑块数是页表宽度。score 末维按这个宽度 16 对齐，超出页表的列保持 -inf。
        page_size = int(key.shape[2] if is_bnbd else key.shape[1])
        max_blocks = int(block_table.shape[1]) if block_table is not None else 1
    score_stride = round_up(max_blocks, SCORE_STRIDE_ALIGN)
    is_quant = np.issubdtype(key.dtype, np.integer)

    if is_quant:
        if scale is None:
            raise ValueError("int8 key requires dequant scale")
    elif scale is not None:
        raise ValueError("non-quant path must not pass scale")

    q = query.astype(np.float32)
    deq = None
    if is_quant:
        deq = scale.astype(np.float32)
        if is_tnd:
            deq = deq.reshape(-1) if deq.ndim == 1 else deq[:, 0]
        else:
            deq = deq[:, 0, :]

    out = np.full((num_q_heads, total_q, score_stride), NEG_INF, dtype=np.float32)

    for b in range(batch):
        q_begin, q_end = int(actual_seq_qlen[b]), int(actual_seq_qlen[b + 1])
        q_len = q_end - q_begin
        if is_tnd:
            cu_k = int(actual_seq_klen[b])
            kv = int(actual_seq_klen[b + 1]) - cu_k
        else:
            cu_k = 0
            kv = int(actual_seq_klen[b])
        num_blocks = (kv + pool_block_size - 1) // pool_block_size if kv > 0 else 0
        if not is_tnd and block_table is not None:
            num_blocks = min(num_blocks, int(block_table.shape[1]))
        q_block = int(start_loc[b])
        local_start = max(0, q_block + 1 - int(local_blocks))

        for t in range(q_begin, q_end):
            t_off = t - q_begin
            visible_key_end = _visible_key_end(sparse_mode, kv, q_len, t_off)
            for blk in range(num_blocks):
                key_lo = blk * pool_block_size
                valid = min(visible_key_end - key_lo, pool_block_size)
                if valid <= 0:
                    continue
                if is_tnd:
                    tok = cu_k + key_lo
                    k_page = k[tok : tok + valid, :]
                    if deq is not None:
                        k_page = k_page * deq[tok : tok + valid, None]
                else:
                    # 存储页表按页粒度。页号超出页表宽度的块保持 -inf。
                    page_idx = key_lo // page_size
                    if page_idx >= block_table.shape[1]:
                        continue
                    page = int(block_table[b, page_idx])
                    off = key_lo % page_size
                    k_page = k[page, off : off + valid, :]
                    if deq is not None:
                        k_page = k_page * deq[page, off : off + valid, None]
                s = q[t] @ k_page.T
                if pool_block_size == 16:
                    # 先在每个 head 内做 block maxpool，再对该 block 取 head 间 max。
                    # 归约结果只留在 head 0，返回前丢掉其余 head。
                    out[0, t, blk] = s.max(axis=1).max()
                else:
                    out[:, t, blk] = s.max(axis=1)

            total_rows = q_len * num_q_heads
            for h in range(num_q_heads):
                visible_end_blk = _tile_visible_end_blk(
                    t_off * num_q_heads + h,
                    total_rows,
                    num_q_heads,
                    sparse_mode,
                    kv,
                    q_len,
                    pool_block_size,
                    num_blocks,
                )
                for blk in range(num_blocks):
                    if blk >= visible_end_blk:
                        continue
                    boost = None
                    if blk < int(init_blocks):
                        boost = LOCAL_SCORE_INIT
                    if local_start <= blk <= q_block:
                        boost = LOCAL_SCORE_LOCAL
                    if boost is not None and (pool_block_size != 16 or h == 0):
                        dst_h = 0 if pool_block_size == 16 else h
                        out[dst_h, t, blk] = boost

    if pool_block_size == 16:
        return out[:1].copy()
    return out


def compare(actual, golden, atol=1e-3, rtol=1e-3, error_ratio_threshold=1e-3):
    """比对 actual 与 golden。填充位必须同为 -inf；强制高分位允许同为 +inf/1e29/1e30。"""
    actual = np.asarray(actual, dtype=np.float32)
    golden = np.asarray(golden, dtype=np.float32)
    if actual.shape != golden.shape:
        return False, f"shape mismatch: actual={actual.shape} golden={golden.shape}"

    fill_a = np.isneginf(actual)
    fill_g = np.isneginf(golden)
    if not np.array_equal(fill_a, fill_g):
        return False, "fill(-inf) mask mismatch"

    # 强制高分：两侧都极大即可
    boost_g = golden >= 1.0e28
    boost_a = actual >= 1.0e28
    if not np.array_equal(boost_a, boost_g):
        return False, "local_mask boost mask mismatch"

    valid = (~fill_g) & (~boost_g)
    if not np.any(valid):
        return True, "ok (all fill/boost)"

    diff = np.abs(actual[valid] - golden[valid])
    denom = np.maximum(np.abs(golden[valid]), 1.0)
    bad = diff > (atol + rtol * denom)
    ratio = float(np.mean(bad)) if bad.size else 0.0
    if ratio > error_ratio_threshold:
        max_diff = float(np.max(diff)) if diff.size else 0.0
        return False, f"error_ratio={ratio:.6f} max_abs_diff={max_diff}"
    return True, "ok"
