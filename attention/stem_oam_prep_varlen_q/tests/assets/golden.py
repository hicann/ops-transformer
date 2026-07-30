#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""
Golden reference variants for stem_oam_prep_varlen_q.

Three implementations are provided, all returning a BF16 numpy array with
shape [batch, H_q, max_Qb, S*D]:

  golden_seq                 - Sequential sum, pure numpy (math baseline).
  golden_torch_vectorized    - Sequential sum, PyTorch vectorized on GPU
                               (no Python loops over batch/head/qb).
                               Mathematically identical to golden_seq.
                               Requires PyTorch.
  golden_binary_reduce       - Backward reduce algorithm, numpy. Mirrors the
                               current kernel implementation exactly:
                                 Phase 1: Muls each row by its scale
                                 Phase 2: backward reduce over R axis;
                                          stride from R_main (largest power
                                          of 2 <= R) down to 1, each round
                                          adds row[r+stride] into row[r].
                               Expected FP32 difference vs golden_seq < 1e-6.

Switch between them by uncommenting exactly one SELECTED_GOLDEN line below.
"""

import math
import numpy as np
import ml_dtypes
import tbetoolkits
from .registry import register_golden


# ============================================================
# Golden variants
# ============================================================


def golden_seq(context: "tbetoolkits.UniversalTestcaseStructure"):
    """
    Sequential sum, numpy + for-loop (math baseline).
    """
    q_np = context.input_arrays[0]
    qscale_np = context.input_arrays[1] if len(context.input_arrays) > 1 else None
    q_seq_lens_np = context.other_runtime_params["qSeqLens"]
    cu_seqlens_q_np = context.other_runtime_params["cuSeqLensQ"]

    B = context.other_compilation_params["stemBlockSize"]
    S = context.other_compilation_params["stemStride"]

    D = q_np.shape[-1]
    H_q = q_np.shape[1]
    R = B // S
    batch = len(q_seq_lens_np)

    print(f"[golden_seq] B={B}, S={S}, D={D}, H_q={H_q}, batch={batch}")
    print(f"[golden_seq] q_seq_lens={q_seq_lens_np}")
    print(f"[golden_seq] cu_seqlens_q={cu_seqlens_q_np}")
    print(
        f"[golden_seq] qscale_shape={qscale_np.shape if qscale_np is not None else None}"
    )

    q_fp32 = q_np.astype(np.float32)
    # 空 qScale (size 0) 视为未提供 scale, 走 unweighted path
    if qscale_np is not None and qscale_np.size > 0:
        qscale = qscale_np.astype(np.float32)
        if qscale.ndim == 1:
            qscale = qscale.reshape(-1, 1)
    else:
        qscale = None

    if batch == 0:
        return np.zeros((0, H_q, 0, S * D), dtype=np.float32).astype(ml_dtypes.bfloat16)

    max_Qb = max(math.ceil(int(sl) / B) for sl in q_seq_lens_np)

    qflat = np.zeros((batch, H_q, max_Qb, S * D), dtype=np.float32)

    for b in range(batch):
        q_len = q_seq_lens_np[b]
        cu_off = cu_seqlens_q_np[b]
        q_padded = math.ceil(q_len / B) * B
        num_Qb = q_padded // B

        if num_Qb == 0:
            continue

        Q_segment = q_fp32[cu_off : cu_off + q_len]
        actual_len = Q_segment.shape[0]
        Q_dense = np.zeros((q_padded, H_q, D), dtype=np.float32)
        Q_dense[:actual_len] = Q_segment

        for h in range(H_q):
            Q_blocks = Q_dense[: num_Qb * B, h, :].reshape(num_Qb, R, S, D)

            if qscale is not None:
                positions = np.arange(num_Qb * B).reshape(num_Qb, R, S)
                # scale 有效性由 q_len 决定 (与 kernel CopyInScalesBulk 语义一致:
                #   只读 min(B, q_len - start_row) 个 scale, 多余 Duplicate 为 0)
                valid_mask = positions < q_len
                scales = np.zeros((num_Qb, R, S), dtype=np.float32)
                scales[valid_mask] = qscale[cu_off + positions[valid_mask], h]
                Q_weighted = Q_blocks * scales[:, :, :, np.newaxis]
            else:
                Q_weighted = Q_blocks

            Q_group_sum = Q_weighted.sum(axis=1)

            for qb in range(num_Qb):
                qflat[b, h, qb, :] = Q_group_sum[qb].reshape(S * D)

    return qflat.astype(ml_dtypes.bfloat16)


def golden_torch_vectorized(context: "tbetoolkits.UniversalTestcaseStructure"):
    """
    Sequential sum, PyTorch vectorized (no Python loop over batch/head/qb).

    Mathematically equivalent to golden_seq; runs on GPU if CUDA is available.
    """
    import torch

    q_np = context.input_arrays[0]
    qscale_np = context.input_arrays[1] if len(context.input_arrays) > 1 else None
    q_seq_lens_np = context.other_runtime_params["qSeqLens"]
    cu_seqlens_q_np = context.other_runtime_params["cuSeqLensQ"]

    B = context.other_compilation_params["stemBlockSize"]
    S = context.other_compilation_params["stemStride"]

    D = q_np.shape[-1]
    H_q = q_np.shape[1]
    R = B // S
    batch = len(q_seq_lens_np)

    print(f"[golden_torch] B={B}, S={S}, D={D}, H_q={H_q}, batch={batch}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    q = torch.from_numpy(q_np.astype(np.float32)).to(device)
    if qscale_np is not None:
        q_t = torch.from_numpy(
            qscale_np.astype(np.float32).reshape(-1, 1)
            if qscale_np.ndim == 1
            else qscale_np.astype(np.float32)
        ).to(device)
    else:
        q_t = None
    q_len_t = torch.tensor(q_seq_lens_np, dtype=torch.int64, device=device)
    cu_off_t = torch.tensor(cu_seqlens_q_np[:-1], dtype=torch.int64, device=device)

    if batch == 0:
        return np.zeros((0, H_q, 0, S * D), dtype=np.float32).astype(ml_dtypes.bfloat16)

    q_padded_lens = ((q_len_t + B - 1) // B) * B
    max_Qb = int(q_padded_lens.max().item()) // B
    max_q_padded = max_Qb * B

    if max_Qb == 0:
        return np.zeros((batch, H_q, 0, S * D), dtype=np.float32).astype(
            ml_dtypes.bfloat16
        )

    # Global positions + valid mask (handles q_len not aligned to B)
    positions = torch.arange(max_q_padded, device=device).unsqueeze(
        0
    )  # [1, max_q_padded]
    global_pos = cu_off_t.unsqueeze(1) + positions  # [batch, max_q_padded]
    valid_mask = positions < q_len_t.unsqueeze(1)  # [batch, max_q_padded]
    gp_safe = torch.where(valid_mask, global_pos, torch.zeros_like(global_pos))

    # De-page Q + reshape to [batch, H_q, max_Qb, R, S, D]
    Q_padded = q[gp_safe]  # [batch, max_q_padded, H_q, D]
    Q_padded = torch.where(
        valid_mask.unsqueeze(-1).unsqueeze(-1), Q_padded, torch.zeros_like(Q_padded)
    )
    Q_blocks = Q_padded.permute(0, 2, 1, 3).reshape(batch, H_q, max_Qb, R, S, D)

    # Scale fusion + weighted group sum along R
    if q_t is not None:
        scale_padded = q_t[gp_safe]  # [batch, max_q_padded, H_q]
        # 用 where 显式清零无效位置, 避免 NaN * 0 = NaN 污染 (kernel Duplicate(0) padding 语义)
        scale_padded = torch.where(
            valid_mask.unsqueeze(-1), scale_padded, torch.zeros_like(scale_padded)
        )
        scales = scale_padded.permute(0, 2, 1).reshape(batch, H_q, max_Qb, R, S)
        Q_weighted = Q_blocks * scales.unsqueeze(-1)
        Q_group_sum = Q_weighted.sum(dim=3)  # [batch, H_q, max_Qb, S, D]
    else:
        Q_group_sum = Q_blocks.sum(dim=3)

    qflat = Q_group_sum.reshape(batch, H_q, max_Qb, S * D).to(torch.bfloat16)
    return qflat.cpu().float().numpy().astype(ml_dtypes.bfloat16)


def golden_binary_reduce(context: "tbetoolkits.UniversalTestcaseStructure"):
    """
    Backward reduce algorithm, numpy (matches current kernel exactly).

    Per block (qb) at a given head h, working on Q_q[r, g, :] with shape
    [R, S, D]:

      Phase 1: Muls each row by its scale
               Q_q[r, g, :] *= scales[r, g]

      Phase 2: backward reduce over R axis.
               R_main = largest power of 2 <= R
               R_tail = R - R_main
               stride from R_main down to 1 (halving each round):
                 ops = R_tail if stride==R_main else stride
                 for r in 0..ops-1: row[r] += row[r + stride]
               After the loop, row[0, g, :] holds the full group sum.
    """
    q_np = context.input_arrays[0]
    qscale_np = context.input_arrays[1] if len(context.input_arrays) > 1 else None
    q_seq_lens_np = context.other_runtime_params["qSeqLens"]
    cu_seqlens_q_np = context.other_runtime_params["cuSeqLensQ"]

    B = context.other_compilation_params["stemBlockSize"]
    S = context.other_compilation_params["stemStride"]

    D = q_np.shape[-1]
    H_q = q_np.shape[1]
    R = B // S
    batch = len(q_seq_lens_np)

    print(f"[golden_binary] B={B}, S={S}, D={D}, H_q={H_q}, batch={batch}")
    print(f"[golden_binary] q_seq_lens={q_seq_lens_np}")
    print(f"[golden_binary] cu_seqlens_q={cu_seqlens_q_np}")
    print(
        f"[golden_binary] qscale_shape={qscale_np.shape if qscale_np is not None else None}"
    )

    q_fp32 = q_np.astype(np.float32)
    # 空 qScale (size 0) 视为未提供 scale, 走 unweighted path
    if qscale_np is not None and qscale_np.size > 0:
        qscale = qscale_np.astype(np.float32)
        if qscale.ndim == 1:
            qscale = qscale.reshape(-1, 1)
    else:
        qscale = None

    if batch == 0:
        return np.zeros((0, H_q, 0, S * D), dtype=np.float32).astype(ml_dtypes.bfloat16)

    max_Qb = max(math.ceil(int(sl) / B) for sl in q_seq_lens_np)

    qflat = np.zeros((batch, H_q, max_Qb, S * D), dtype=np.float32)

    for b in range(batch):
        q_len = q_seq_lens_np[b]
        cu_off = cu_seqlens_q_np[b]
        q_padded = math.ceil(q_len / B) * B
        num_Qb = q_padded // B

        if num_Qb == 0:
            continue

        Q_segment = q_fp32[cu_off : cu_off + q_len]
        Q_dense = np.zeros((q_padded, H_q, D), dtype=np.float32)
        Q_dense[: Q_segment.shape[0]] = Q_segment

        for h in range(H_q):
            if qscale is not None:
                positions = np.arange(num_Qb * B).reshape(num_Qb, R, S)
                # scale 有效性由 q_len 决定 (与 kernel CopyInScalesBulk 语义一致)
                valid_mask = positions < q_len
                scales = np.zeros((num_Qb, R, S), dtype=np.float32)
                scales[valid_mask] = qscale[cu_off + positions[valid_mask], h]
            else:
                scales = None

            for qb in range(num_Qb):
                # Q_q: [R, S, D] — this block, this head
                Q_q = Q_dense[qb * B : (qb + 1) * B, h, :].reshape(R, S, D).copy()

                # Phase 1: Muls each (r, g) row by scales[r, g]
                if scales is not None:
                    Q_q = Q_q * scales[qb, :, :, np.newaxis]

                R_main = 1
                while (R_main << 1) <= R:
                    R_main <<= 1
                R_tail = R - R_main

                stride = R_main
                while stride >= 1:
                    ops = R_tail if stride == R_main else stride
                    if ops > 0:
                        for r in range(ops):
                            Q_q[r] += Q_q[r + stride]
                    stride >>= 1

                # row[0, g, :] now holds groupSum[g, :]
                qflat[b, h, qb, :] = Q_q[0].reshape(S * D)

    return qflat.astype(ml_dtypes.bfloat16)


# ============================================================
# Switch: uncomment exactly ONE of the three lines below.
# ============================================================
# SELECTED_GOLDEN = golden_seq
# SELECTED_GOLDEN = golden_torch_vectorized    # requires PyTorch
SELECTED_GOLDEN = golden_binary_reduce  # matches current kernel algorithm


# ============================================================
# Unified entry point (registered under the operator name)
# ============================================================
@register_golden(["stem_oam_prep_varlen_q"])
def stem_oam_prep_varlen_q(context: "tbetoolkits.UniversalTestcaseStructure"):
    return SELECTED_GOLDEN(context)
