# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Reference and precision harness for sparse_lightning_indexer_kl_loss_grad.

TND, N2=1 flow:
  1. Recompute main-attention sparse softmax on CPU to produce attn_softmax_l1_norm.
  2. Pass attn_softmax_l1_norm into the fused op.
  3. Compare dq, dk, dw, and softmax_out with the CPU reference.
"""

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch_npu


def find_repo_root():
    for path in Path(__file__).resolve().parents:
        if (path / "MindSpeed").exists() or (path / "build.sh").exists():
            return path
    return Path(__file__).resolve().parent


REPO_ROOT = find_repo_root()

MINDSPEED_PATH = REPO_ROOT / "MindSpeed"
if MINDSPEED_PATH.exists():
    sys.path.insert(0, str(MINDSPEED_PATH))

import cann_ops_transformer  # noqa: E402,F401  pylint: disable=wrong-import-position,unused-import


SLI_METADATA_SIZE = 64
SLI_METADATA_MAX_CORE_NUM = 25
SLI_META_CORE_NUM_INDEX = 0
SLI_META_TOTAL_SIZE_INDEX = 1
SLI_META_SPLIT_FACTOR_SIZE_INDEX = 2
SLI_META_BS1_INDEX_BASE = 8


def get_cu_seqlens(seqlens_list):
    seqlens_list = normalize_lengths(seqlens_list)
    cu = torch.zeros(len(seqlens_list) + 1, dtype=torch.int64)
    for i in range(len(seqlens_list) + 1):
        cu[i] = sum(seqlens_list[:i])
    return cu


def normalize_lengths(lengths):
    if isinstance(lengths, int):
        return (lengths,)
    return tuple(lengths)


def cumulative(lengths):
    lengths = normalize_lengths(lengths)
    total, out = 0, []
    for length in lengths:
        total += length
        out.append(total)
    return tuple(out)


def normalize_residuals(residuals, batch_size):
    if isinstance(residuals, int):
        return (residuals,) * batch_size
    residuals = tuple(residuals)
    if len(residuals) != batch_size:
        raise ValueError(
            f"cmp_residual_k batch mismatch: {residuals}, batch_size={batch_size}"
        )
    return residuals


def dtype_from_name(name):
    name = name.lower()
    if name in ("fp16", "float16"):
        return torch.float16
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp32", "float32", "float"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def print_slig_metadata(metadata):
    print("========= metadata output =========")
    print(
        f"device={metadata.device}, dtype={metadata.dtype}, shape={tuple(metadata.shape)}"
    )
    meta_cpu = metadata.detach().cpu().view(-1)
    if meta_cpu.numel() < SLI_META_BS1_INDEX_BASE + SLI_METADATA_MAX_CORE_NUM:
        print(f"metadata length too short: {meta_cpu.numel()}")
        return

    core_num = int(meta_cpu[SLI_META_CORE_NUM_INDEX].item())
    total_size = int(meta_cpu[SLI_META_TOTAL_SIZE_INDEX].item())
    split_factor_size = int(meta_cpu[SLI_META_SPLIT_FACTOR_SIZE_INDEX].item())
    reserved = [int(meta_cpu[i].item()) for i in range(3, SLI_META_BS1_INDEX_BASE)]
    b_s1_index = [
        int(meta_cpu[SLI_META_BS1_INDEX_BASE + i].item())
        for i in range(SLI_METADATA_MAX_CORE_NUM)
    ]

    print(
        f"coreNum={core_num}, totalSize={total_size}, splitFactorSize={split_factor_size}"
    )
    print(f"reserved={reserved}")
    print(f"bS1Index all={b_s1_index}")
    if core_num > 0:
        print(f"bS1Index active={b_s1_index[:core_num]}")
    print("===================================")


def trunc_div(numerator, denominator):
    if numerator >= 0:
        return numerator // denominator
    return -((-numerator) // denominator)


def valid_k_count(local_q_idx, kv_len, q_len, cmp_ratio, cmp_residual_k=0):
    if cmp_ratio != 0:
        pre_compress_k_len = kv_len * cmp_ratio + cmp_residual_k
        numerator = pre_compress_k_len - q_len + local_q_idx + 1
        return trunc_div(numerator, cmp_ratio)
    return max(kv_len - q_len + local_q_idx + 1, 0)


def make_invalid_mask(q_len, kv_len, topk, cmp_ratio, cmp_residual_k=0):
    mask = torch.zeros(q_len, topk, dtype=torch.bool)
    for q_idx in range(q_len):
        real_k = valid_k_count(q_idx, kv_len, q_len, cmp_ratio, cmp_residual_k)
        real_k = max(0, min(real_k, topk))
        if real_k < topk:
            mask[q_idx, real_k:] = True
    return mask


@dataclass(frozen=True)
class TndCase:
    name: str = "Test"
    q_lens: tuple = (128, 16, 8)
    kv_lens: tuple = (32, 4, 2)
    n_query: int = 128
    n_query_index: int = 64
    n_key: int = 1
    d: int = 512
    d_index: int = 128
    topk: int = 512
    dtype: str = "bf16"
    sparse_mode: int = 3
    layout: str = "TND"
    cmp_residual_k: tuple = 0
    attn_l1_scale: float = 1.0

    @property
    def q_lens_tuple(self):
        return normalize_lengths(self.q_lens)

    @property
    def kv_lens_tuple(self):
        return normalize_lengths(self.kv_lens)

    @property
    def batch_size(self):
        if len(self.q_lens_tuple) != len(self.kv_lens_tuple):
            raise ValueError(
                f"q_lens and kv_lens batch mismatch: {self.q_lens_tuple} vs {self.kv_lens_tuple}"
            )
        return len(self.q_lens_tuple)

    @property
    def cmp_residual_tuple(self):
        return normalize_residuals(self.cmp_residual_k, self.batch_size)

    @property
    def total_q(self):
        return sum(self.q_lens_tuple)

    @property
    def total_kv(self):
        return sum(self.kv_lens_tuple)

    @property
    def scale(self):
        return 1 / math.sqrt(self.d)

    @property
    def cu_q(self):
        return cumulative(self.q_lens_tuple)

    @property
    def cu_kv(self):
        return cumulative(self.kv_lens_tuple)


DEFAULT_CASES = (
    TndCase(name="single_tiny_c4", q_lens=(16,), kv_lens=(4,), attn_l1_scale=0.39),
    TndCase(name="single_mid_c4", q_lens=(512,), kv_lens=(128,), attn_l1_scale=0.55),
    TndCase(name="multi_base_c4", q_lens=(128, 16, 8), kv_lens=(32, 4, 2)),
    TndCase(name="multi_unaligned_c4", q_lens=(20, 36, 68), kv_lens=(5, 9, 17)),
    TndCase(name="multi_mixed_c4", q_lens=(4, 8, 32, 128), kv_lens=(1, 2, 8, 32)),
    TndCase(name="single_residual_c4", q_lens=(18,), kv_lens=(4,), cmp_residual_k=(2,)),
    TndCase(name="single_int_lengths_c4", q_lens=128, kv_lens=32),
    TndCase(name="single_kv_extra", q_lens=(128,), kv_lens=(128,)),
    TndCase(name="multi_kv_extra", q_lens=(16, 64, 128), kv_lens=(8, 32, 128)),
    TndCase(
        name="single_scaled_l1_c4", q_lens=(512,), kv_lens=(128,), attn_l1_scale=0.37
    ),
)


def list_cases(cases):
    print("available cases:")
    for idx, case in enumerate(cases):
        print(
            f"  {idx}: {case.name}, q_lens={case.q_lens_tuple}, "
            f"kv_lens={case.kv_lens_tuple}, cmp_residual_k={case.cmp_residual_tuple}, "
            f"attn_l1_scale={case.attn_l1_scale}"
        )


def select_cases(case_selector, cases):
    if case_selector == "all":
        return list(cases)

    if case_selector.isdigit():
        idx = int(case_selector)
        if idx < 0 or idx >= len(cases):
            raise ValueError(f"case index out of range: {idx}, total={len(cases)}")
        return [cases[idx]]

    matched = [case for case in cases if case.name == case_selector]
    if not matched:
        names = ", ".join(case.name for case in cases)
        raise ValueError(
            f"unknown case '{case_selector}', available: all, 0..{len(cases) - 1}, {names}"
        )
    return matched


def make_tensor(shape, dtype, seed, low=None, high=None):
    generator = torch.Generator().manual_seed(seed)
    tensor = torch.rand(*shape, generator=generator, dtype=dtype)
    if low is not None and high is not None:
        tensor = tensor * (high - low) + low
    return tensor


def make_sparse_indices(case, cmp_ratio):
    sparse_indices = torch.full(
        (case.total_q, case.n_key, case.topk), -1, dtype=torch.int32
    )
    q_start = 0
    for q_len, kv_len, cmp_residual_k in zip(
        case.q_lens_tuple, case.kv_lens_tuple, case.cmp_residual_tuple
    ):
        for local_q_idx in range(q_len):
            real_k = valid_k_count(
                local_q_idx, kv_len, q_len, cmp_ratio, cmp_residual_k
            )
            if real_k <= 0:
                continue
            used_k = min(real_k, case.topk)
            sparse_indices[q_start + local_q_idx, 0, :used_k] = torch.randperm(
                real_k, dtype=torch.int32
            )[:used_k]
        q_start += q_len
    return sparse_indices


def make_inputs(case, cmp_ratio):
    dtype = dtype_from_name(case.dtype)
    q = make_tensor((case.total_q, case.n_query, case.d), dtype, 42)
    k = make_tensor((case.total_kv, case.n_key, case.d), dtype, 43)
    q_index = make_tensor((case.total_q, case.n_query_index, case.d_index), dtype, 44)
    k_index = make_tensor((case.total_kv, case.n_key, case.d_index), dtype, 45)
    weights = make_tensor((case.total_q, case.n_query_index), torch.float32, 48) * (
        0.1 / 6.0
    )
    return {
        "q": q,
        "k": k,
        "q_index": q_index,
        "k_index": k_index,
        "weights": weights,
        "sparse_indices": make_sparse_indices(case, cmp_ratio),
    }


def make_cu_seqlens(lengths, device):
    lengths = normalize_lengths(lengths)
    cu = torch.zeros(len(lengths) + 1, dtype=torch.int32, device=device)
    cu[1:] = torch.tensor(lengths, dtype=torch.int32, device=device).cumsum(0)
    return cu


class SparseLightningIndexerKLLossGradRef:
    def __init__(
        self,
        q,
        k,
        q_index,
        k_index,
        weights,
        sparse_indices,
        actual_q_len,
        actual_kv_len,
        cmp_residual_k=0,
        cmp_ratio=4,
        scale=None,
        attn_l1_scale=1.0,
        dtype_internal=torch.float32,
    ):
        assert k.shape[1] == 1, "Only N2=1 is supported."
        self.q = q
        self.k = k
        self.q_index = q_index
        self.k_index = k_index
        self.weights = weights
        self.sparse_indices = sparse_indices
        self.actual_q_len = list(actual_q_len)
        self.actual_kv_len = list(actual_kv_len)
        self.cmp_residual_k = list(
            normalize_residuals(cmp_residual_k, len(self.actual_q_len))
        )
        self.cu_q = get_cu_seqlens(self.actual_q_len)
        self.cu_kv = get_cu_seqlens(self.actual_kv_len)
        self.cmp_ratio = cmp_ratio
        self.scale = scale if scale is not None else 1.0 / math.sqrt(q.shape[-1])
        self.attn_l1_scale = float(attn_l1_scale)
        self.input_dtype = q_index.dtype
        self.dtype_internal = dtype_internal

    def forward(self):
        total_q, _, _ = self.q.shape
        total_kv, nk, _ = self.k.shape
        _, n_query, _ = self.q.shape
        _, n_query_index, d_index = self.q_index.shape

        topk = self.sparse_indices.shape[-1]
        group = n_query // nk
        group_index = n_query_index // nk
        dt = self.dtype_internal
        in_dt = self.input_dtype

        attn_softmax_l1_norm = torch.zeros((total_q, nk, topk), dtype=dt)
        softmax_out = torch.zeros((total_q, nk, topk), dtype=dt)
        d_query_index = torch.zeros((total_q, n_query_index, d_index), dtype=dt)
        d_key_index = torch.zeros((total_kv, nk, d_index), dtype=dt)
        d_weights = torch.zeros((total_q, n_query_index), dtype=dt)

        for b_idx in range(len(self.actual_q_len)):
            qs = self.cu_q[b_idx].item()
            qe = self.cu_q[b_idx + 1].item()
            ks = self.cu_kv[b_idx].item()
            ke = self.cu_kv[b_idx + 1].item()
            q_len = qe - qs
            kv_len = ke - ks
            cmp_residual_k = self.cmp_residual_k[b_idx]

            q_b = self.q[qs:qe].to(dt)
            k_b = self.k[ks:ke].to(dt)
            q_index_b = self.q_index[qs:qe].to(dt)
            k_index_b = self.k_index[ks:ke].to(dt)
            weights_b = self.weights[qs:qe].to(dt)
            sparse_b = self.sparse_indices[qs:qe].to(torch.int64)

            invalid_row = (sparse_b == -1).all(dim=-1)
            sparse = sparse_b.squeeze(1)
            valid = (sparse >= 0) & (sparse < kv_len)
            safe_id = sparse.clamp(0, kv_len - 1)
            invalid_topk_mask = make_invalid_mask(
                q_len, kv_len, topk, self.cmp_ratio, cmp_residual_k
            )

            k_main = k_b.squeeze(1)
            k_topk = k_main[safe_id].masked_fill(~valid.unsqueeze(-1), 0.0)
            qk = torch.einsum("thd,tsd->ths", q_b, k_topk) * self.scale
            qk = qk.masked_fill(invalid_topk_mask.unsqueeze(1), float("-inf"))
            p_full = nn.functional.softmax(qk, dim=-1)
            p = p_full.reshape(q_len, nk, group, topk).sum(dim=2) / group
            p = p.masked_fill(invalid_topk_mask.unsqueeze(1), 0.0)
            p = p.masked_fill(invalid_row.unsqueeze(-1), 0.0)
            if self.attn_l1_scale != 1.0:
                p = p * self.attn_l1_scale

            k_index_main = k_index_b.squeeze(1)
            k_index_topk = k_index_main[safe_id].masked_fill(~valid.unsqueeze(-1), 0.0)
            s_logits = torch.einsum("thd,tsd->ths", q_index_b, k_index_topk)
            relu_s = torch.relu(s_logits)
            s_weighted = relu_s * weights_b.unsqueeze(-1)
            s_reduce = s_weighted.reshape(q_len, nk, group_index, topk).sum(dim=2)
            s_reduce = s_reduce.masked_fill(
                invalid_topk_mask.unsqueeze(1), float("-inf")
            )
            s_softmax = nn.functional.softmax(s_reduce, dim=-1)
            s_softmax = s_softmax.masked_fill(invalid_topk_mask.unsqueeze(1), 0.0)
            s_softmax = s_softmax.masked_fill(invalid_row.unsqueeze(-1), 0.0)

            p_reduce = p.sum(dim=-1, keepdim=True)
            ds_reduce = s_softmax * p_reduce - p
            ds_sq = ds_reduce.squeeze(1)

            dw = torch.einsum("tjs,ts->tj", relu_s, ds_sq)
            dw = dw.masked_fill(invalid_row, 0.0)

            d_s_logits = (
                ds_sq.unsqueeze(1) * weights_b.unsqueeze(-1) * (relu_s > 0).to(dt)
            )
            d_s_logits_lp = d_s_logits.to(in_dt).to(dt)

            dqi = torch.einsum("tjs,tsd->tjd", d_s_logits_lp, k_index_topk)
            dqi = dqi.masked_fill(invalid_row.unsqueeze(-1), 0.0)

            dki_gather = torch.einsum("tjs,tjd->tsd", d_s_logits_lp, q_index_b)
            dki_gather = dki_gather.masked_fill(invalid_row.unsqueeze(-1), 0.0)
            dki_gather = dki_gather * valid.unsqueeze(-1)

            dki_per_batch = torch.zeros(kv_len, d_index, dtype=dt)
            dki_per_batch.index_add_(
                0,
                safe_id.reshape(-1),
                dki_gather.reshape(-1, d_index),
            )

            attn_softmax_l1_norm[qs:qe] = p
            softmax_out[qs:qe] = s_softmax
            d_query_index[qs:qe] = dqi
            d_key_index[ks:ke, 0, :] = dki_per_batch
            d_weights[qs:qe] = dw

        return {
            "attn_softmax_l1_norm": attn_softmax_l1_norm,
            "d_query_index": d_query_index,
            "d_key_index": d_key_index,
            "d_weights": d_weights,
            "softmax_out": softmax_out,
        }


def run_golden(case, tensors, cmp_ratio):
    print(f"========= run_golden: cmp_ratio={cmp_ratio}, dtype={case.dtype} =========")
    ref = SparseLightningIndexerKLLossGradRef(
        q=tensors["q"],
        k=tensors["k"],
        q_index=tensors["q_index"],
        k_index=tensors["k_index"],
        weights=tensors["weights"],
        sparse_indices=tensors["sparse_indices"],
        actual_q_len=case.q_lens_tuple,
        actual_kv_len=case.kv_lens_tuple,
        cmp_residual_k=case.cmp_residual_tuple,
        cmp_ratio=cmp_ratio,
        scale=case.scale,
        attn_l1_scale=case.attn_l1_scale,
        dtype_internal=torch.float32,
    )
    out = ref.forward()
    print("golden outputs:", {key: tuple(value.shape) for key, value in out.items()})
    return out


def run_fused_op(case, tensors, golden, device, cmp_ratio):
    cu_seqlens_q = make_cu_seqlens(case.q_lens_tuple, device)
    cu_seqlens_k = make_cu_seqlens(case.kv_lens_tuple, device)
    cmp_residual_k = torch.tensor(
        case.cmp_residual_tuple, dtype=torch.int32, device=device
    )

    q = tensors["q_index"].contiguous().to(device)
    k = tensors["k_index"].contiguous().to(device)
    w = tensors["weights"].to(torch.float32).contiguous().to(device)
    sparse_indices = tensors["sparse_indices"].contiguous().to(device)
    attn_softmax_l1_norm = golden["attn_softmax_l1_norm"].contiguous().to(device)

    print("##################### START fused inputs #####################")
    print("q(q_index):", q.shape, q.dtype)
    print("k(k_index):", k.shape, k.dtype)
    print("w:", w.shape, w.dtype)
    print("sparse_indices:", sparse_indices.shape, sparse_indices.dtype)
    print(
        "attn_softmax_l1_norm:", attn_softmax_l1_norm.shape, attn_softmax_l1_norm.dtype
    )
    print("cu_seqlens_q:", cu_seqlens_q.tolist())
    print("cu_seqlens_k:", cu_seqlens_k.tolist())
    print("cmp_residual_k:", cmp_residual_k.tolist())
    print(
        "layout:", case.layout, "mask_mode:", case.sparse_mode, "cmp_ratio:", cmp_ratio
    )
    print("##################### END fused inputs #######################")

    metadata = (
        torch.ops.cann_ops_transformer.sparse_lightning_indexer_kl_loss_grad_metadata(
            case.n_query_index,
            case.n_key,
            case.d_index,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            cmp_residual_k=cmp_residual_k,
            batch_size=case.batch_size,
            max_seqlen_q=max(case.q_lens_tuple),
            max_seqlen_k=max(case.kv_lens_tuple),
            topk=case.topk,
            layout_q=case.layout,
            layout_k=case.layout,
            mask_mode=case.sparse_mode,
            cmp_ratio=cmp_ratio,
        )
    )
    print_slig_metadata(metadata)

    dq, dk, dw, softmax_out = (
        torch.ops.cann_ops_transformer.sparse_lightning_indexer_kl_loss_grad(
            q,
            k,
            w,
            sparse_indices,
            attn_softmax_l1_norm,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            cmp_residual_k=cmp_residual_k,
            metadata=metadata,
            layout_q=case.layout,
            layout_k=case.layout,
            mask_mode=case.sparse_mode,
            cmp_ratio=cmp_ratio,
        )
    )
    torch.npu.synchronize()
    return {
        "d_query_index": dq.cpu(),
        "d_key_index": dk.cpu(),
        "d_weights": dw.cpu(),
        "softmax_out": softmax_out.cpu(),
    }


def relative_diff(actual, expected, diff_thd):
    abs_diff = torch.abs(actual - expected)
    base = torch.maximum(torch.abs(actual), torch.abs(expected))
    min_base = float((1.0 / (1 << 14)) / diff_thd)
    return torch.where(
        abs_diff < diff_thd,
        abs_diff,
        abs_diff / (torch.maximum(base, torch.full_like(base, min_base)) + 1e-9),
    )


def compare_tensors(
    name, actual, expected, diff_thd=0.005, pct_thd=0.005, max_diff_thd=0.1
):
    actual_flat = actual.float().contiguous().view(-1)
    expected_flat = expected.float().contiguous().view(-1)
    if actual_flat.numel() != expected_flat.numel():
        print(
            f"{name}: failed, shape mismatch a={actual_flat.shape} e={expected_flat.shape}"
        )
        return {"name": name, "result": "failed", "pct": 0.0, "max_error": float("inf")}

    rel = relative_diff(actual_flat, expected_flat, diff_thd)
    errors = rel[rel > diff_thd]
    pct = (actual_flat.numel() - errors.numel()) / max(actual_flat.numel(), 1) * 100.0
    max_error = float(errors.max().item()) if errors.numel() else 0.0
    max_abs = (
        float(torch.abs(actual_flat - expected_flat).max().item())
        if actual_flat.numel()
        else 0.0
    )
    result = (
        "success"
        if pct >= (1 - pct_thd) * 100.0 and max_error < max_diff_thd
        else "failed"
    )
    print(
        f"{name}: {result}, pct={pct:.6f}%, max_abs_diff={max_abs:.8f}, max_error={max_error}"
    )
    return {"name": name, "result": result, "pct": pct, "max_error": max_error}


def print_case_summary(case):
    print("\n[1/4] Input case")
    print(
        f"case={case.name}, layout={case.layout}, dtype={case.dtype}, "
        f"q_lens={case.q_lens_tuple}, kv_lens={case.kv_lens_tuple}, "
        f"cmp_residual_k={case.cmp_residual_tuple}, attn_l1_scale={case.attn_l1_scale}"
    )
    print(
        f"Tq={case.total_q}, Tkv={case.total_kv}, "
        f"Nq={case.n_query}, Nqi={case.n_query_index}, topK={case.topk}"
    )
    print(f"cu_q={case.cu_q}, cu_kv={case.cu_kv}")


def run_one_case(case, device, cmp_ratio, detail_compare):
    inputs = make_inputs(case, cmp_ratio)
    print_case_summary(case)

    print("\n[2/4] Generate CPU reference")
    golden = run_golden(case, inputs, cmp_ratio)

    print("\n[3/4] Run fused op")
    fused = run_fused_op(case, inputs, golden, device, cmp_ratio)
    print("fused outputs:", {key: tuple(value.shape) for key, value in fused.items()})

    print("\n[4/4] Compare")
    results = [
        compare_tensors(
            "d_query_index", fused["d_query_index"], golden["d_query_index"]
        ),
        compare_tensors("d_key_index", fused["d_key_index"], golden["d_key_index"]),
        compare_tensors(
            "d_weights",
            fused["d_weights"],
            golden["d_weights"],
            diff_thd=0.05,
            pct_thd=0.05,
        ),
        compare_tensors("softmax_out", fused["softmax_out"], golden["softmax_out"]),
    ]

    if detail_compare:
        try:
            from precision_compare import data_compare

            data_compare(
                fused["d_query_index"].float().numpy(),
                golden["d_query_index"].float().numpy(),
            )
            data_compare(
                fused["d_key_index"].float().numpy(),
                golden["d_key_index"].float().numpy(),
            )
            data_compare(
                fused["d_weights"].float().numpy(), golden["d_weights"].float().numpy()
            )
            data_compare(
                fused["softmax_out"].float().numpy(),
                golden["softmax_out"].float().numpy(),
            )
        except ImportError:
            print("precision_compare is not available; skipped detailed data_compare.")

    print("\ncase compare summary:")
    for item in results:
        print(
            f"{item['name']}: {item['result']}, pct={item['pct']:.6f}%, max_error={item['max_error']}"
        )
    return results


def make_cp_q_ranges(total_q, cp_parts, random_splits=False, seed=2026, min_first=1):
    if cp_parts <= 0:
        raise ValueError(f"cp_parts must be positive, got {cp_parts}")
    if cp_parts == 1:
        return [(0, total_q)]

    if random_splits:
        generator = torch.Generator().manual_seed(seed)
        candidates = torch.arange(max(1, min_first), total_q, dtype=torch.int64)
        perm = candidates[torch.randperm(candidates.numel(), generator=generator)]
        points = sorted(int(x.item()) for x in perm[: cp_parts - 1])
    else:
        points = [round(total_q * idx / cp_parts) for idx in range(1, cp_parts)]

    points = sorted(set(max(1, min(total_q - 1, point)) for point in points))
    boundaries = [0] + points + [total_q]
    return [
        (boundaries[idx], boundaries[idx + 1]) for idx in range(len(boundaries) - 1)
    ]


def make_cp_shard_case(q_start, q_end, cmp_ratio, attn_l1_scale=1.0):
    pre_compress_k_len = q_end
    kv_len = pre_compress_k_len // cmp_ratio
    cmp_residual_k = pre_compress_k_len % cmp_ratio
    if kv_len <= 0:
        raise ValueError(
            f"CP shard [{q_start}, {q_end}) maps to empty compressed KV. "
            f"Use fewer parts or larger first shard."
        )
    return TndCase(
        name=f"cp_q_{q_start}_{q_end}_kv_0_{kv_len}",
        q_lens=(q_end - q_start,),
        kv_lens=(kv_len,),
        cmp_residual_k=(cmp_residual_k,),
        attn_l1_scale=attn_l1_scale,
    )


def slice_cp_inputs(inputs, q_start, q_end, kv_end):
    return {
        "q": inputs["q"][q_start:q_end].contiguous(),
        "k": inputs["k"][:kv_end].contiguous(),
        "q_index": inputs["q_index"][q_start:q_end].contiguous(),
        "k_index": inputs["k_index"][:kv_end].contiguous(),
        "weights": inputs["weights"][q_start:q_end].contiguous(),
        "sparse_indices": inputs["sparse_indices"][q_start:q_end].contiguous(),
    }


def run_cp_validation(device, cmp_ratio, cp_parts=4, random_splits=False, cp_seed=2026):
    print("\n================ CP validation ================")
    if cmp_ratio <= 0:
        raise ValueError(f"CP validation requires positive cmp_ratio, got {cmp_ratio}")
    ori_q_len = 512
    full_kv_len = ori_q_len // cmp_ratio
    full_residual = ori_q_len % cmp_ratio
    full_case = TndCase(
        name=f"cp_full_q{ori_q_len}_kv{full_kv_len}",
        q_lens=(ori_q_len,),
        kv_lens=(full_kv_len,),
        cmp_residual_k=(full_residual,),
        attn_l1_scale=0.37,
    )
    full_inputs = make_inputs(full_case, cmp_ratio)
    cp_ranges = make_cp_q_ranges(
        full_case.total_q, cp_parts, random_splits, cp_seed, min_first=cmp_ratio
    )

    print("[CP 1/5] Full CPU reference")
    print_case_summary(full_case)
    full_golden = run_golden(full_case, full_inputs, cmp_ratio)

    print("[CP 2/5] Full fused op without CP")
    full_fused = run_fused_op(full_case, full_inputs, full_golden, device, cmp_ratio)
    full_checks = [
        compare_tensors(
            "cp_full_d_query_index",
            full_fused["d_query_index"],
            full_golden["d_query_index"],
        ),
        compare_tensors(
            "cp_full_d_key_index", full_fused["d_key_index"], full_golden["d_key_index"]
        ),
        compare_tensors(
            "cp_full_d_weights",
            full_fused["d_weights"],
            full_golden["d_weights"],
            diff_thd=0.05,
            pct_thd=0.05,
        ),
        compare_tensors(
            "cp_full_softmax_out", full_fused["softmax_out"], full_golden["softmax_out"]
        ),
    ]

    cp_dq_parts = []
    cp_dw_parts = []
    cp_softmax_parts = []
    cp_dk_accum = torch.zeros_like(full_fused["d_key_index"])
    shard_checks = []

    print("[CP 3/5] Run fused op per CP shard")
    print(f"CP ranges={cp_ranges}, random_splits={random_splits}, seed={cp_seed}")
    for shard_idx, (q_start, q_end) in enumerate(cp_ranges):
        cp_case = make_cp_shard_case(q_start, q_end, cmp_ratio, full_case.attn_l1_scale)
        kv_end = cp_case.kv_lens_tuple[0]
        cp_inputs = slice_cp_inputs(full_inputs, q_start, q_end, kv_end)
        print(
            f"\n---- CP shard {shard_idx}: q=[{q_start}, {q_end}), kv=[0, {kv_end}) ----"
        )
        print_case_summary(cp_case)
        cp_golden = run_golden(cp_case, cp_inputs, cmp_ratio)
        cp_fused = run_fused_op(cp_case, cp_inputs, cp_golden, device, cmp_ratio)

        shard_checks.extend(
            [
                compare_tensors(
                    f"cp_shard_{shard_idx}_d_query_index",
                    cp_fused["d_query_index"],
                    cp_golden["d_query_index"],
                ),
                compare_tensors(
                    f"cp_shard_{shard_idx}_d_key_index",
                    cp_fused["d_key_index"],
                    cp_golden["d_key_index"],
                ),
                compare_tensors(
                    f"cp_shard_{shard_idx}_d_weights",
                    cp_fused["d_weights"],
                    cp_golden["d_weights"],
                    diff_thd=0.05,
                    pct_thd=0.05,
                ),
                compare_tensors(
                    f"cp_shard_{shard_idx}_softmax_out",
                    cp_fused["softmax_out"],
                    cp_golden["softmax_out"],
                ),
            ]
        )

        cp_dq_parts.append(cp_fused["d_query_index"])
        cp_dw_parts.append(cp_fused["d_weights"])
        cp_softmax_parts.append(cp_fused["softmax_out"])
        cp_dk_accum[:kv_end] += cp_fused["d_key_index"]

    cp_merged = {
        "d_query_index": torch.cat(cp_dq_parts, dim=0),
        "d_key_index": cp_dk_accum,
        "d_weights": torch.cat(cp_dw_parts, dim=0),
        "softmax_out": torch.cat(cp_softmax_parts, dim=0),
    }

    print("[CP 4/5] Compare merged CP outputs with full fused outputs")
    merged_vs_full_checks = [
        compare_tensors(
            "cp_vs_full_d_query_index",
            cp_merged["d_query_index"],
            full_fused["d_query_index"],
        ),
        compare_tensors(
            "cp_vs_full_d_key_index",
            cp_merged["d_key_index"],
            full_fused["d_key_index"],
            diff_thd=0.05,
            pct_thd=0.05,
        ),
        compare_tensors(
            "cp_vs_full_d_weights",
            cp_merged["d_weights"],
            full_fused["d_weights"],
            diff_thd=0.05,
            pct_thd=0.05,
        ),
        compare_tensors(
            "cp_vs_full_softmax_out",
            cp_merged["softmax_out"],
            full_fused["softmax_out"],
        ),
    ]

    print("[CP 5/5] Compare merged CP outputs with full CPU reference")
    merged_vs_golden_checks = [
        compare_tensors(
            "cp_vs_golden_d_query_index",
            cp_merged["d_query_index"],
            full_golden["d_query_index"],
        ),
        compare_tensors(
            "cp_vs_golden_d_key_index",
            cp_merged["d_key_index"],
            full_golden["d_key_index"],
        ),
        compare_tensors(
            "cp_vs_golden_d_weights",
            cp_merged["d_weights"],
            full_golden["d_weights"],
            diff_thd=0.05,
            pct_thd=0.05,
        ),
        compare_tensors(
            "cp_vs_golden_softmax_out",
            cp_merged["softmax_out"],
            full_golden["softmax_out"],
        ),
    ]

    return (
        full_case,
        full_checks + shard_checks + merged_vs_full_checks + merged_vs_golden_checks,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--cmp-ratio", type=int, default=4)
    parser.add_argument(
        "--case",
        default="all",
        help="Case name/index to run. Use 'all' to run the default sequence-length suite.",
    )
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--detail-compare", action="store_true")
    parser.add_argument("--only-cp-validation", action="store_true")
    parser.add_argument("--skip-cp-validation", action="store_true")
    parser.add_argument("--cp-parts", type=int, default=4)
    parser.add_argument("--cp-random-splits", action="store_true")
    parser.add_argument("--cp-seed", type=int, default=2026)
    args = parser.parse_args()

    if args.list_cases:
        list_cases(DEFAULT_CASES)
        return

    torch_npu.npu.set_device(args.device_id)
    device = torch.device(f"npu:{args.device_id}")

    all_results = []
    if not args.only_cp_validation:
        cases = select_cases(args.case, DEFAULT_CASES)
        detail_compare = args.detail_compare or len(cases) == 1
        for case_idx, case in enumerate(cases):
            print(
                f"\n================ Case {case_idx + 1}/{len(cases)}: {case.name} ================"
            )
            results = run_one_case(case, device, args.cmp_ratio, detail_compare)
            all_results.append((case, results))

    if args.only_cp_validation or not args.skip_cp_validation:
        cp_case, cp_results = run_cp_validation(
            device,
            args.cmp_ratio,
            cp_parts=args.cp_parts,
            random_splits=args.cp_random_splits,
            cp_seed=args.cp_seed,
        )
        all_results.append((cp_case, cp_results))

    print("\n================ suite summary ================")
    failed = False
    for case, results in all_results:
        case_ok = all(item["result"] == "success" for item in results)
        failed = failed or not case_ok
        status = "success" if case_ok else "failed"
        detail = ", ".join(f"{item['name']}={item['result']}" for item in results)
        print(f"{case.name}: {status} ({detail})")
    if failed:
        raise RuntimeError("One or more cases failed.")


if __name__ == "__main__":
    main()
