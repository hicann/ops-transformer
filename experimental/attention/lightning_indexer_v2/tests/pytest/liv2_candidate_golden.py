#!/usr/bin/python
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

"""CPU golden for the LightningIndexerV2 candidate (source / candidate_mode==1) path.

Provenance
----------
Implements the reference described in
``candidate-source impl design`` §1.1 / §5.1 and the semantic baseline
``two-level topk design`` §1.1 (QLI v2 ``select_candidate_blocks``).

This module is deliberately dependency-light (numpy + torch only, NO torch_npu /
cann_ops_transformer_custom_li import) so that the golden and the comparator can be exercised
on a pure-CPU host before the device kernel exists.  ``lightning_indexer_v2_golden``
re-exports these helpers so that ``lightning_indexer_v2_golden.select_candidate_blocks_ref``
matches the file location named in the design's implementation checklist (§6).

Formalization (per batch b, query row i, N2 head n):
    score(b,i,:)    fp32 index_score, -inf at unreachable positions
    blockSize       candidate_block_size in [2,64], power of two (default 8)
    actS2Size(b)    compressed valid KV length of batch b
    numBlocks(b)    ceil(actS2Size(b) / blockSize)
    blkScore(b,i,j) max_{blockSize*j <= p < min(blockSize*(j+1), actS2Size)} score(b,i,p)
    rowValidLen(b,i) reachable prefix length of row (b,i)  (= kernel cuRealAcSeq)
    lastBlk(b,i)    (rowValidLen(b,i) - 1) // blockSize
    blkScore(b,i,lastBlk) = +inf                            # pin
    candidate_topk_indices(b,i,:) = blkScore descending top candidate_topk_blocks block ids
                                    (output width == candidate_topk_blocks; blkScore==-inf -> -1)
"""

import numpy as np
import torch

NEG_INF = float("-inf")
POS_INF = float("inf")

# candidate_topk_blocks sentinel: -1 == off (fall back to现网 behaviour).
CANDIDATE_TOPK_BLOCKS_OFF = -1
CANDIDATE_BLOCK_SIZE_DEFAULT = 8


def _as_2d_score(score):
    """Return (flat[R, S2] float32 ndarray, original leading shape)."""
    arr = np.asarray(score, dtype=np.float32)
    if arr.ndim < 1:
        raise ValueError(
            f"score must have at least one dimension, got shape {arr.shape}"
        )
    width = arr.shape[-1]
    lead = arr.shape[:-1]
    if width == 0:
        # reshape(-1, 0) is ambiguous for size-0 arrays; rows == product of leading dims.
        rows = int(np.prod(lead)) if lead else 1
        return np.zeros((rows, 0), dtype=np.float32), lead
    flat = arr.reshape(-1, width)
    return flat, lead


def _broadcast_compress_lens(compress_lens, rows):
    """Normalize compress_lens into an int64 [rows] vector.

    Supports the two forms required by the design (§5.1):
      * scalar (mask_mode=0, batch level)
      * row-level vector / [S1,1] (mask_mode=3)
    """
    if compress_lens is None:
        raise ValueError("compress_lens must not be None")
    if np.isscalar(compress_lens):
        return np.full(rows, int(compress_lens), dtype=np.int64)
    cl = np.asarray(compress_lens).reshape(-1).astype(np.int64)
    if cl.size == 1:
        return np.full(rows, int(cl[0]), dtype=np.int64)
    if cl.size != rows:
        raise ValueError(
            f"compress_lens length {cl.size} does not match flattened row count {rows}"
        )
    return cl


def derive_compress_lens(score):
    """Row-level reachable length (rowValidLen) derived from the -inf mask.

    ``score`` is [..., S2] with -inf at unreachable positions.  Because both
    mask_mode=0 (no mask) and mask_mode=3 (causal right-down) leave a contiguous
    reachable prefix ``[0, rowValidLen)``, the reachable length equals the count of
    finite entries per row.  Returns an int64 array shaped like ``score.shape[:-1]``.
    """
    arr = np.asarray(score, dtype=np.float32)
    finite = np.isfinite(arr) & (arr > NEG_INF)
    return finite.sum(axis=-1).astype(np.int64)


def select_candidate_blocks_ref(score, compress_lens, topk_blocks, block_size):
    """numpy row-by-row reconstruction of the candidate block selection (ground truth).

    Args:
        score: [..., S2] fp32 index_score with -inf at unreachable positions.  The last
            dimension must be the batch's compressed valid length actS2Size (or wider with
            -inf padding; trailing all -inf blocks simply become -1 slots).
        compress_lens: scalar (mask_mode=0) or row-level vector broadcastable to the
            flattened leading dims (mask_mode=3).  Value == rowValidLen per row.
        topk_blocks: candidate_topk_blocks, the fixed output width.
        block_size: candidate_block_size in [2,64], power of two.

    Returns:
        int32 ndarray [..., topk_blocks]; selected block ids (unordered), -1 for empty slots.
    """
    if topk_blocks <= 0:
        raise ValueError(f"topk_blocks must be > 0, got {topk_blocks}")
    if block_size < 1:
        raise ValueError(f"block_size must be >= 1, got {block_size}")

    flat, lead = _as_2d_score(score)
    rows, width = flat.shape
    cl_flat = _broadcast_compress_lens(compress_lens, rows)

    num_blocks = (width + block_size - 1) // block_size
    pad = num_blocks * block_size - width
    out = np.full((rows, topk_blocks), -1, dtype=np.int32)
    if num_blocks == 0:
        return out.reshape(lead + (topk_blocks,))

    take = min(topk_blocks, num_blocks)
    for r in range(rows):
        row = flat[r]
        if pad > 0:
            row = np.concatenate([row, np.full(pad, NEG_INF, dtype=np.float32)])
        blk = row.reshape(num_blocks, block_size).max(axis=1).astype(np.float32)
        vl = int(cl_flat[r])
        if vl > 0:
            last = (vl - 1) // block_size
            if 0 <= last < num_blocks:
                blk[last] = POS_INF  # pin: latest token block无条件保留
        # descending, stable (ties keep ascending block id) -> deterministic golden
        order = np.argsort(-blk, kind="stable")
        sel = order[:take]
        sel_scores = blk[sel]
        valid = sel_scores > NEG_INF
        idx = np.where(valid, sel, -1).astype(np.int32)
        out[r, :take] = idx
    return out.reshape(lead + (topk_blocks,))


def select_candidate_blocks_torch(score, compress_lens, topk_blocks, block_size):
    """torch implementation mirroring the model reference (cross-validation oracle).

    Direct transcription of ``select_candidate_blocks`` in
    ``two-level topk design`` §1.1:
        F.pad(..., -inf) -> unflatten(...).amax(-1) -> pin last -> topk -> scatter(>-inf).
    Tie-break order may differ from the numpy reference; the comparison contract is
    SET-based (see result_compare_method.check_result_candidate), so this is a valid
    independent oracle for the selected block SET and the -1 slot count.
    """
    import torch.nn.functional as F

    if topk_blocks <= 0:
        raise ValueError(f"topk_blocks must be > 0, got {topk_blocks}")
    if block_size < 1:
        raise ValueError(f"block_size must be >= 1, got {block_size}")

    t = torch.as_tensor(np.asarray(score, dtype=np.float32), dtype=torch.float32)
    lead = t.shape[:-1]
    width = t.shape[-1]
    if width == 0:
        return torch.full(lead + (topk_blocks,), -1, dtype=torch.int32)
    flat = t.reshape(-1, width)
    rows = flat.shape[0]

    cl = _broadcast_compress_lens(compress_lens, rows)
    cl_t = torch.from_numpy(cl)

    num_blocks = (width + block_size - 1) // block_size
    out = torch.full((rows, topk_blocks), -1, dtype=torch.int32)
    if num_blocks == 0:
        return out.reshape(lead + (topk_blocks,))

    pad = num_blocks * block_size - width
    if pad > 0:
        flat = F.pad(flat, (0, pad), value=NEG_INF)
    blk = flat.unflatten(-1, (num_blocks, block_size)).amax(-1)  # [R, numBlocks]

    valid_pin = cl_t > 0
    last = torch.where(valid_pin, (cl_t - 1) // block_size, torch.zeros_like(cl_t))
    last = last.clamp(min=0, max=num_blocks - 1)
    row_idx = torch.arange(rows)
    blk[row_idx[valid_pin], last[valid_pin]] = POS_INF

    take = min(topk_blocks, num_blocks)
    topv, topi = blk.topk(take, dim=-1)
    keep = topv > NEG_INF
    placed = torch.where(
        keep, topi.to(torch.int32), torch.full_like(topi, -1, dtype=torch.int32)
    )
    out[:, :take] = placed
    return out.reshape(lead + (topk_blocks,))


def block_scores_from_score(score, block_size):
    """Per-row block amax matrix (before pin), used for boundary-tolerance arbitration.

    Args:
        score: [..., S2] fp32 with -inf at unreachable positions.  The last dim must be
            exactly the batch's actS2Size (caller slices per batch).
        block_size: candidate_block_size.
    Returns:
        float32 ndarray [..., numBlocks] with the pin already applied at lastBlk when the
        row is reachable (rowValidLen>0).  numBlocks == ceil(S2 / block_size).
    """
    arr = np.asarray(score, dtype=np.float32)
    width = arr.shape[-1]
    lead = arr.shape[:-1]
    num_blocks = (width + block_size - 1) // block_size
    if num_blocks == 0:
        return np.zeros(lead + (0,), dtype=np.float32)
    pad = num_blocks * block_size - width
    flat = arr.reshape(-1, width)
    if pad > 0:
        flat = np.concatenate(
            [flat, np.full((flat.shape[0], pad), NEG_INF, dtype=np.float32)], axis=1
        )
    blk = (
        flat.reshape(flat.shape[0], num_blocks, block_size)
        .max(axis=2)
        .astype(np.float32)
    )
    # rowValidLen from the un-padded score (finite count over the real width)
    finite = np.isfinite(arr) & (arr > NEG_INF)
    vl = finite.reshape(-1, width).sum(axis=-1).astype(np.int64)
    rows = blk.shape[0]
    for r in range(rows):
        if vl[r] > 0:
            last = (int(vl[r]) - 1) // block_size
            if 0 <= last < num_blocks:
                blk[r, last] = POS_INF
    return blk.reshape(lead + (num_blocks,))


def compute_candidate_golden_bnsd(
    score_bnsd,
    topk_blocks,
    block_size,
    return_block_scores=False,
):
    """Build the candidate golden in BNSD layout from the fp32 score matrix.

    Args:
        score_bnsd: [B, N2, S1, S2max] fp32 index_score (the ``topk_value`` returned by
            ``GeneralizedLIV2.forward``); -inf at unreachable / padding positions.
        topk_blocks: candidate_topk_blocks (output width).
        block_size: candidate_block_size.
        return_block_scores: also return the per-row block-amax matrix (for tie arbitration).

    Returns:
        cand_bnsd: int32 [B, N2, S1, topk_blocks] block ids (-1 slots), and optionally
        blk_bnsd: float32 [B, N2, S1, NBmax] block scores (-inf padded across batches).

    actS2Size(b) and rowValidLen(b,i) are derived from the score itself:
        rowValidLen(b,i) = finite count of score[b,:,i,:]
        actS2Size(b)     = max_i rowValidLen(b,i)
    This is faithful because the causal / batch mask is already baked into the score.
    """
    arr = np.asarray(score_bnsd, dtype=np.float32)
    if arr.ndim != 4:
        raise ValueError(f"score_bnsd must be [B,N2,S1,S2], got shape {arr.shape}")
    B, N2, S1, S2max = arr.shape

    if topk_blocks == CANDIDATE_TOPK_BLOCKS_OFF:
        cand = np.full((B, N2, S1, 0), -1, dtype=np.int32)
        if return_block_scores:
            return cand, np.zeros((B, N2, S1, 0), dtype=np.float32)
        return cand

    # per-batch actS2Size = max reachable count over (n2, i)
    finite = np.isfinite(arr) & (arr > NEG_INF)
    row_valid = finite.sum(axis=-1)  # [B, N2, S1]
    act_s2 = row_valid.reshape(B, -1).max(axis=1) if B > 0 else np.zeros((0,), np.int64)
    act_s2 = np.asarray(act_s2, dtype=np.int64)

    nb_per_batch = (act_s2 + block_size - 1) // block_size
    nb_max = int(nb_per_batch.max()) if nb_per_batch.size else 0

    cand = np.full((B, N2, S1, topk_blocks), -1, dtype=np.int32)
    blk = (
        np.full((B, N2, S1, nb_max), NEG_INF, dtype=np.float32)
        if return_block_scores
        else None
    )

    for b in range(B):
        s2 = int(act_s2[b])
        if s2 <= 0:
            continue  # whole batch empty -> all -1 (matches CleanInvalidOutput)
        nb = int(nb_per_batch[b])
        for n in range(N2):
            for i in range(S1):
                row = arr[b, n, i, :s2]
                vl = int(row_valid[b, n, i])
                cand[b, n, i, :] = select_candidate_blocks_ref(
                    row, vl, topk_blocks, block_size
                )
                if return_block_scores:
                    bs = block_scores_from_score(row, block_size)
                    blk[b, n, i, : bs.shape[-1]] = bs
    if return_block_scores:
        return cand, blk
    return cand


def bnsd_to_output_layout(tensor_bnsd, layout_q, cu_seqlens_q=None, fill=-1):
    """Transform a BNSD [B,N2,S1,D] int/float tensor to the operator output layout.

    Mirrors ``GeneralizedLIV2.trans_bnsd_to_layout`` for the candidate output:
      * BSND -> [B, S1, N2, D] (permute 0,2,1,3)
      * TND  -> [T, N2, D] packed by cu_seqlens_q (per-batch reachable S1 prefix)
    ``cu_seqlens_q`` may be the prefix-sum form (size B+1) or per-batch lengths (size B).
    ``fill`` is used for TND padding rows / untouched slots.
    """
    arr = np.asarray(tensor_bnsd)
    B, N2, S1, D = arr.shape
    if layout_q == "BSND":
        return np.ascontiguousarray(arr.transpose(0, 2, 1, 3))
    if layout_q in ("TND", "TND_NTD"):
        if cu_seqlens_q is None:
            raise ValueError(
                "TND layout requires cu_seqlens_q (per-batch reachable length)"
            )
        act = np.asarray(cu_seqlens_q).reshape(-1).astype(np.int64)
        if act.size == B + 1:  # prefix-sum form [0, l0, l0+l1, ...]
            act = np.diff(act)
        T = int(act.sum())
        out = np.full((T, N2, D), fill, dtype=arr.dtype)
        t = 0
        for b in range(B):
            n_rows = int(act[b])
            if n_rows <= 0:
                continue
            out[t : t + n_rows] = arr[b, :, :n_rows, :].transpose(1, 0, 2)
            t += n_rows
        if layout_q == "TND_NTD":
            out = np.ascontiguousarray(out.transpose(1, 0, 2))
        return out
    raise ValueError(f"unsupported layout_q for candidate output: {layout_q}")
