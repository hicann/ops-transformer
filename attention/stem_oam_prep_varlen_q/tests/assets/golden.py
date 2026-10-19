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
import re
from types import SimpleNamespace

import numpy as np
import ml_dtypes

try:
    import tbetoolkits
except ImportError:  # standalone self-check (`python3 golden.py`) has no tbetoolkits
    tbetoolkits = None
try:
    from .registry import register_golden
except ImportError:  # standalone self-check has no package context

    def register_golden(names):
        def decorator(func):
            return func

        return decorator


# DYN_GOLD platform loading contracts (module level, same style as
# causal_conv1d / mhc_pre_sinkhorn_backward). No aclnn golden is needed for
# this operator.
__golden__ = {"kernel": {"stem_oam_prep_varlen_q": "stem_oam_prep_varlen_q_golden"}}

__input__ = {"kernel": {"stem_oam_prep_varlen_q": "StemOamPrepVarlenQ_input"}}


# ============================================================
# Golden variants
# ============================================================


def _extract_inputs(context: "tbetoolkits.UniversalTestcaseStructure"):
    """
    Extract (q, qScale, qSeqLens, cuSeqLensQ) from the test context.

    Preferred 4-tensor contract, matching the op def input order
    (q -> qSeqLens -> cuSeqLensQ -> qScale(OPTIONAL)):
      input_arrays[0]=q, [1]=qSeqLens, [2]=cuSeqLensQ, [3]=qScale.
    qScale stays optional: with len(input_arrays) == 3 it is None (unweighted
    path). Falls back to the legacy 2-tensor flow (input_arrays[1]=qScale,
    qSeqLens/cuSeqLensQ from other_runtime_params) when fewer than 3 tensor
    inputs are provided.
    """
    arrays = context.input_arrays
    q_np = arrays[0]
    if len(arrays) >= 3:
        q_seq_lens_np = arrays[1]
        cu_seqlens_q_np = arrays[2]
        qscale_np = arrays[3] if len(arrays) > 3 else None
    else:
        qscale_np = arrays[1] if len(arrays) > 1 else None
        q_seq_lens_np = context.other_runtime_params["qSeqLens"]
        cu_seqlens_q_np = context.other_runtime_params["cuSeqLensQ"]
    return q_np, qscale_np, q_seq_lens_np, cu_seqlens_q_np


def golden_seq(context: "tbetoolkits.UniversalTestcaseStructure"):
    """
    Sequential sum, numpy + for-loop (math baseline).
    """
    q_np, qscale_np, q_seq_lens_np, cu_seqlens_q_np = _extract_inputs(context)

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

    q_np, qscale_np, q_seq_lens_np, cu_seqlens_q_np = _extract_inputs(context)

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
    q_np, qscale_np, q_seq_lens_np, cu_seqlens_q_np = _extract_inputs(context)

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


# ============================================================
# DYN_GOLD platform entry points (referenced by __golden__/__input__)
# ============================================================
def stem_oam_prep_varlen_q_golden(*input_arrays, **kwargs):
    """Platform golden entry: positional tensor array -> golden output.

    Wraps the platform-provided inputs into a context-shaped object so the
    existing golden variants keep their signature. With the 4-tensor contract
    (q, qSeqLens, cuSeqLensQ, qScale) the tensor-input branch of
    _extract_inputs is taken naturally; stemBlockSize/stemStride come from the
    CSV attributes relayed via kwargs (code defaults as fallback).
    """
    context = SimpleNamespace(
        input_arrays=list(input_arrays),
        other_runtime_params=kwargs,
        other_compilation_params={
            "stemBlockSize": kwargs.get("stemBlockSize", 128),
            "stemStride": kwargs.get("stemStride", 16),
        },
    )
    return SELECTED_GOLDEN(context)


def StemOamPrepVarlenQ_input(*input_arrays, **kwargs):
    """Custom input generator: rebuild the varlen index inputs.

    Input order: q(0) -> qSeqLens(1) -> cuSeqLensQ(2) -> qScale(3).

    Platform-pregenerated int64 data cannot satisfy the varlen prefix-sum
    contract, so both index inputs are rebuilt here:
      - qSeqLens: authoritative values come from the CSV attributes relayed
        via kwargs['qSeqLens']; otherwise parsed from the testcase name
        suffix `_lens<val(_val)*>` (e.g. _lens0_128 -> [0, 128]), same
        precedence as causal_conv1d's CausalConv1d_input; without either
        source the tokens are split evenly across batches (remainder to the
        last batch, sum strictly == total_tokens). Every source must pass
        the shape validation (len == batch, sum == total_tokens, each >= 0),
        otherwise the next source is tried.
      - cuSeqLensQ: always rebuilt as [0, cumsum(qSeqLens)]; platform values
        are never trusted.
      - q(0) and qScale(3) carry no semantic constraint and stay untouched.
    """
    input_arrays = list(input_arrays)
    if len(input_arrays) < 3 or input_arrays[1] is None:
        return input_arrays

    q = input_arrays[0]
    qsl = input_arrays[1]
    total_tokens = int(q.shape[0]) if q is not None and np.ndim(q) == 3 else 0
    batch = int(qsl.shape[0]) if np.ndim(qsl) > 0 else 0
    if batch <= 0:
        return input_arrays

    def _validated(values):
        if (
            len(values) == batch
            and sum(values) == total_tokens
            and all(v >= 0 for v in values)
        ):
            return values
        return None

    # a) CSV attributes relayed by the platform (most authoritative)
    seq_lens = None
    design = kwargs.get("qSeqLens")
    if design is not None:
        seq_lens = _validated([int(v) for v in design])
    # b) design values encoded in the testcase name (_lens256_300_128)
    if seq_lens is None:
        match = re.search(r"_lens(\d+(?:_\d+)*)", kwargs.get("testcase_name", ""))
        if match:
            seq_lens = _validated([int(v) for v in match.group(1).split("_")])
    # c) even-split fallback (remainder to the last batch)
    if seq_lens is None:
        base, remainder = divmod(total_tokens, batch)
        seq_lens = [base] * batch
        seq_lens[-1] += remainder

    input_arrays[1] = np.array(seq_lens, dtype=np.int64)
    cu_seqlens = [0]
    for seq_len in seq_lens:
        cu_seqlens.append(cu_seqlens[-1] + seq_len)
    input_arrays[2] = np.array(cu_seqlens, dtype=np.int64)
    return input_arrays


# ============================================================
# Standalone self-check: `python3 golden.py`
# Verifies the golden variants stay consistent under both the
# 4-tensor input contract and the legacy 2-tensor fallback flow.
# ============================================================
if __name__ == "__main__":
    import sys

    class _StubContext:
        """Minimal duck-typed stand-in for UniversalTestcaseStructure."""

        def __init__(self, input_arrays, runtime_params, compilation_params):
            self.input_arrays = input_arrays
            self.other_runtime_params = runtime_params
            self.other_compilation_params = compilation_params

    def _build_context(num_heads, q_seq_lens, four_tensor):
        B = 128
        S = 16
        batch = len(q_seq_lens)
        total_tokens = sum(q_seq_lens)
        cu_seqlens = [0]
        for seq_len in q_seq_lens:
            cu_seqlens.append(cu_seqlens[-1] + seq_len)

        # Exact-representable data (small integers, power-of-two scales) so
        # every partial sum is exact in FP32 under any reduce order.
        rng = np.random.default_rng(7)
        q_np = rng.integers(-4, 5, size=(total_tokens, num_heads, 128))
        q_np = q_np.astype(np.float32).astype(ml_dtypes.float8_e4m3fn)
        qscale_np = rng.choice([0.5, 1.0, 2.0], size=(total_tokens, num_heads)).astype(
            np.float32
        )

        runtime = {"qSeqLens": list(q_seq_lens), "cuSeqLensQ": cu_seqlens}
        comp = {"stemBlockSize": B, "stemStride": S}
        if four_tensor:
            # op def order: q, qSeqLens, cuSeqLensQ, qScale(OPTIONAL)
            arrays = [
                q_np,
                np.array(q_seq_lens, dtype=np.int64),
                np.array(cu_seqlens, dtype=np.int64),
                qscale_np,
            ]
        else:
            # legacy flow: q, qScale + runtime params
            arrays = [q_np, qscale_np]
        return _StubContext(arrays, runtime, comp)

    def _max_abs_err(a, b):
        return float(np.max(np.abs(a.astype(np.float32) - b.astype(np.float32))))

    cases = [
        ("b1h2_len128_aligned", 2, [128]),
        ("b3h2_multiblock_maxqb3", 2, [256, 300, 128]),
        ("b2h8_zerolen_batch", 8, [0, 128]),
        ("b1h4_len100_tail", 4, [100]),
    ]
    TOL = 1.137e-13
    failed = False
    for label, four_tensor in [("4-input", True), ("legacy-2-input-fallback", False)]:
        for name, num_heads, q_seq_lens in cases:
            ctx = _build_context(num_heads, q_seq_lens, four_tensor)
            out_seq = golden_seq(ctx)
            out_sel = SELECTED_GOLDEN(ctx)
            err_sel = _max_abs_err(out_seq, out_sel)
            print(
                f"[{label}] {name}: out{out_sel.shape} "
                f"seq-vs-selected logic_err={err_sel:.3e}"
            )
            failed = failed or err_sel > TOL
            try:
                out_torch = golden_torch_vectorized(ctx)
            except ImportError:
                print(f"[{label}] {name}: torch unavailable, skipped")
                continue
            err_torch = _max_abs_err(out_seq, out_torch)
            print(
                f"[{label}] {name}: out{out_torch.shape} "
                f"seq-vs-torch logic_err={err_torch:.3e}"
            )
            failed = failed or err_torch > TOL

    # ---- DYN_GOLD platform flow simulation ----
    # CSV cases: (testcase_name, q_shape, qsl_shape, cu_shape, qs_shape,
    #             out_shape, attributes). Per-case design values live both in
    #             the attributes and in the `_lens<...>` name suffix.
    csv_cases = [
        (
            "stem_oam_prep_varlen_q_st_01_b1h2_len128_aligned_1block_lens128",
            (128, 2, 128),
            (1,),
            (2,),
            (128, 2),
            (1, 2, 1, 2048),
            {
                "stemBlockSize": 128,
                "stemStride": 16,
                "qSeqLens": [128],
                "cuSeqLensQ": [0, 128],
            },
        ),
        (
            "stem_oam_prep_varlen_q_st_02_b2h2_len128_len128_multibatch_lens128_128",
            (256, 2, 128),
            (2,),
            (3,),
            (256, 2),
            (2, 2, 1, 2048),
            {
                "stemBlockSize": 128,
                "stemStride": 16,
                "qSeqLens": [128, 128],
                "cuSeqLensQ": [0, 128, 256],
            },
        ),
        (
            "stem_oam_prep_varlen_q_st_03_b1h4_len100_tail_in_block_lens100",
            (100, 4, 128),
            (1,),
            (2,),
            (100, 4),
            (1, 4, 1, 2048),
            {
                "stemBlockSize": 128,
                "stemStride": 16,
                "qSeqLens": [100],
                "cuSeqLensQ": [0, 100],
            },
        ),
        (
            "stem_oam_prep_varlen_q_st_04_b3h2_multiblock_maxqb3_lens256_300_128",
            (684, 2, 128),
            (3,),
            (4,),
            (684, 2),
            (3, 2, 3, 2048),
            {
                "stemBlockSize": 128,
                "stemStride": 16,
                "qSeqLens": [256, 300, 128],
                "cuSeqLensQ": [0, 256, 556, 684],
            },
        ),
        (
            "stem_oam_prep_varlen_q_st_05_b2h8_zerolen_batch_zerofill_lens0_128",
            (128, 8, 128),
            (2,),
            (3,),
            (128, 8),
            (2, 8, 1, 2048),
            {
                "stemBlockSize": 128,
                "stemStride": 16,
                "qSeqLens": [0, 128],
                "cuSeqLensQ": [0, 0, 128],
            },
        ),
        (
            "stem_oam_prep_varlen_q_st_06_b1h16_len200_multiblock_tail_lens200",
            (200, 16, 128),
            (1,),
            (2,),
            (200, 16),
            (1, 16, 2, 2048),
            {
                "stemBlockSize": 128,
                "stemStride": 16,
                "qSeqLens": [200],
                "cuSeqLensQ": [0, 200],
            },
        ),
    ]

    def _pregenerate(rng, q_shape, qsl_shape, cu_shape, qs_shape, total_tokens):
        # Platform-pregenerated inputs: q as random fp8 bit patterns
        # (NaN codes 0x7F/0xFF masked out), index inputs deliberately wrong.
        raw = rng.integers(0, 256, size=q_shape, dtype=np.uint8)
        raw[raw == 0x7F] = 0x7E
        raw[raw == 0xFF] = 0xFE
        q_pre = raw.view(ml_dtypes.float8_e4m3fn)
        qsl_pre = rng.integers(0, 1024, size=qsl_shape, dtype=np.int64)
        cu_pre = rng.integers(0, total_tokens + 1, size=cu_shape, dtype=np.int64)
        qs_pre = rng.uniform(0.5, 2.0, size=qs_shape).astype(np.float32)
        return q_pre, qsl_pre, cu_pre, qs_pre

    def _assert_prefix_sum(tag, fixed, q_pre, qs_pre, batch, total_tokens):
        q_fix, qsl_fix, cu_fix, qs_fix = fixed
        # prefix-sum contract on the fixed index inputs
        assert q_fix is q_pre and qs_fix is qs_pre, (
            f"{tag}: q/qScale must stay untouched"
        )
        assert qsl_fix.dtype == np.int64 and cu_fix.dtype == np.int64, f"{tag}: dtype"
        assert qsl_fix.shape[0] == batch, f"{tag}: len(qSeqLens) == batch"
        assert int(qsl_fix.sum()) == total_tokens, (
            f"{tag}: sum(qSeqLens) == total_tokens"
        )
        assert bool((qsl_fix >= 0).all()), f"{tag}: qSeqLens >= 0"
        assert int(cu_fix[0]) == 0, f"{tag}: cu[0] == 0"
        assert int(cu_fix[-1]) == total_tokens, f"{tag}: cu[-1] == total_tokens"
        assert all(
            int(cu_fix[i + 1]) - int(cu_fix[i]) == int(qsl_fix[i]) for i in range(batch)
        ), f"{tag}: cu prefix sums"

    def _design_context(q_pre, qs_pre, attrs):
        design_ql = np.array(attrs["qSeqLens"], dtype=np.int64)
        design_cu = np.array(attrs["cuSeqLensQ"], dtype=np.int64)
        ctx = _StubContext(
            [q_pre, design_ql, design_cu, qs_pre],
            {"qSeqLens": list(design_ql), "cuSeqLensQ": list(design_cu)},
            attrs,
        )
        return ctx

    rng = np.random.default_rng(11)
    for name, q_shape, qsl_shape, cu_shape, qs_shape, out_shape, attrs in csv_cases:
        total_tokens = q_shape[0]
        batch = qsl_shape[0]
        design_maxqb = max(math.ceil(v / 128) for v in attrs["qSeqLens"])
        assert out_shape[2] == design_maxqb, f"{name}: CSV maxQb vs design"
        q_pre, qsl_pre, cu_pre, qs_pre = _pregenerate(
            rng, q_shape, qsl_shape, cu_shape, qs_shape, total_tokens
        )

        # Scenario A (board-real): kwargs carries ONLY testcase_name, no
        # attributes -> design values must come from the `_lens...` name suffix.
        fixed_a = StemOamPrepVarlenQ_input(
            q_pre, qsl_pre, cu_pre, qs_pre, testcase_name=name
        )
        _assert_prefix_sum(f"{name} A", fixed_a, q_pre, qs_pre, batch, total_tokens)
        assert fixed_a[1].tolist() == attrs["qSeqLens"], (
            f"{name} A: name-parsed qSeqLens != design {attrs['qSeqLens']}"
        )
        out_a = stem_oam_prep_varlen_q_golden(*fixed_a, testcase_name=name)
        assert out_a.shape == out_shape, (
            f"{name} A: golden shape {out_a.shape} != CSV {out_shape}"
        )
        assert out_a.shape[2] == design_maxqb, f"{name} A: maxQb != {design_maxqb}"
        out_design = SELECTED_GOLDEN(_design_context(q_pre, qs_pre, attrs))
        assert np.array_equal(
            out_a.astype(np.float32), out_design.astype(np.float32)
        ), f"{name} A: golden != design golden"
        print(
            f"[SCN-A name-only] {name}: lens={fixed_a[1].tolist()} "
            f"maxQb={out_a.shape[2]} out{out_a.shape} == CSV {out_shape}"
        )

        # Scenario B: kwargs carries attributes (same values as the name
        # suffix) -> identical result to scenario A.
        fixed_b = StemOamPrepVarlenQ_input(
            q_pre, qsl_pre, cu_pre, qs_pre, **dict(attrs, testcase_name=name)
        )
        _assert_prefix_sum(f"{name} B", fixed_b, q_pre, qs_pre, batch, total_tokens)
        assert fixed_b[1].tolist() == fixed_a[1].tolist(), f"{name} B: qSeqLens differs"
        assert fixed_b[2].tolist() == fixed_a[2].tolist(), f"{name} B: cu differs"
        out_b = stem_oam_prep_varlen_q_golden(
            *fixed_b, **dict(attrs, testcase_name=name)
        )
        assert np.array_equal(out_b.astype(np.float32), out_a.astype(np.float32)), (
            f"{name} B: golden differs from A"
        )
        print(f"[SCN-B with-attrs] {name}: identical to scenario A")

    # Scenario C: neither attributes nor a parseable testcase_name ->
    # even-split fallback (remainder to the last batch). Case st_01:
    # even split [128] keeps maxQb == design (1).
    name, q_shape, qsl_shape, cu_shape, qs_shape, out_shape, attrs = csv_cases[0]
    total_tokens = q_shape[0]
    batch = qsl_shape[0]
    q_pre, qsl_pre, cu_pre, qs_pre = _pregenerate(
        rng, q_shape, qsl_shape, cu_shape, qs_shape, total_tokens
    )
    fixed = StemOamPrepVarlenQ_input(
        q_pre,
        qsl_pre,
        cu_pre,
        qs_pre,
        stemBlockSize=attrs["stemBlockSize"],
        stemStride=attrs["stemStride"],
    )
    even = [total_tokens // batch] * batch
    even[-1] += total_tokens % batch
    _assert_prefix_sum("SCN-C even-split", fixed, q_pre, qs_pre, batch, total_tokens)
    assert fixed[1].tolist() == even, f"SCN-C: expected even split {even}"
    cu_even = [0]
    for v in even:
        cu_even.append(cu_even[-1] + v)
    design_ctx = _StubContext(
        [
            q_pre,
            np.array(even, dtype=np.int64),
            np.array(cu_even, dtype=np.int64),
            qs_pre,
        ],
        {"qSeqLens": even, "cuSeqLensQ": cu_even},
        attrs,
    )
    out_c = stem_oam_prep_varlen_q_golden(
        *fixed, stemBlockSize=attrs["stemBlockSize"], stemStride=attrs["stemStride"]
    )
    out_design_c = SELECTED_GOLDEN(design_ctx)
    assert out_c.shape == out_design_c.shape, "SCN-C: shape mismatch"
    assert np.array_equal(out_c.astype(np.float32), out_design_c.astype(np.float32)), (
        "SCN-C: golden mismatch"
    )
    even_maxqb = max(math.ceil(v / 128) for v in even)
    assert out_c.shape[2] == even_maxqb, "SCN-C: maxQb vs even-split design"
    print(
        f"[SCN-C even-split] {name}: split={even} maxQb={out_c.shape[2]} "
        f"out{out_c.shape}, prefix-sum OK, golden identical"
    )

    if failed:
        print(f"SELF-CHECK FAILED (logic_err > {TOL})")
        sys.exit(1)
    print(f"SELF-CHECK PASSED (logic_err <= {TOL})")
    sys.exit(0)
