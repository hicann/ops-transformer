# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""CPU-only input builders and independent E2E reference for fused MX QKV.

Load with TTK --plugin <this file>. No TTK internals or NPU runtime are needed
to import this module or run the reference. Inputs retain their declared dtype.
"""

import numpy as np
import torch

__spec__ = {
    "torch.ops.cann_ops_transformer.fused_quant_qkv_proj_norm_rope_cache": "FusedQuantQkvProjNormRopeCacheE2eTestSpec",
}


def selected_rows(
    m, tokens_per_rank=None, chunk_tokens_per_rank=None, chunk_token_offset=None
):
    """Map local token rows to the caller's dense metadata/output rows."""
    rows = torch.arange(m, dtype=torch.int64)
    chunk = (tokens_per_rank, chunk_tokens_per_rank, chunk_token_offset)
    if all(value is None for value in chunk):
        return rows
    if any(value is None for value in chunk):
        raise ValueError("All three chunk attributes must be specified together")
    if not (
        0 < chunk_tokens_per_rank <= tokens_per_rank
        and 0 <= chunk_token_offset <= tokens_per_rank - chunk_tokens_per_rank
        and m % chunk_tokens_per_rank == 0
    ):
        raise ValueError("Invalid chunk dimensions or offset")
    return (
        rows // chunk_tokens_per_rank * tokens_per_rank
        + chunk_token_offset
        + rows % chunk_tokens_per_rank
    )


def quantize_mx(values, payload, scale):
    """Quantize finite FP32 values in groups of 32 with positive E8M0 scales.

    Scale is the next power of two at or above max(abs(group))/FP8_max.
    This produces valid paired test data; it is not a benchmark of NPU quantize.
    Write E8M0 bytes explicitly to avoid CPU E8M0 arithmetic dependencies.
    """
    grouped = values.reshape(values.shape[0], values.shape[1] // 32, 32)
    peak = grouped.abs().amax(dim=-1)
    exponent = torch.ceil(
        torch.log2(peak.clamp_min(2.0**-100) / torch.finfo(payload.dtype).max)
    )
    exponent = exponent.clamp(-127, 127)
    factors = torch.exp2(exponent)
    payload.copy_((grouped / factors[..., None]).reshape_as(values).to(payload.dtype))
    scale.view(torch.uint8).copy_((exponent + 127).to(torch.uint8).reshape_as(scale))


def customize_inputs(
    hidden_quant,
    hidden_scale,
    weight_quant,
    weight_scale,
    k_gamma,
    positions,
    cos_sin,
    slot_mapping,
    q_out,
    k_out,
    v_out,
    k_cache,
    v_cache,
    *,
    epsilon=1e-6,
    mirror_k_out=None,
    mirror_v_out=None,
    positions_contiguous=False,
    tokens_per_rank=None,
    chunk_tokens_per_rank=None,
    chunk_token_offset=None,
    **kwargs,
):
    """Construct legal metadata and paired quantized inputs on CPU.

    Distinct sentinels cover untouched dense/chunk rows and untouched cache rows.
    When M exceeds cache capacity, surplus rows use slot=-1 (skip write).
    """
    m, k = hidden_quant.shape
    n = weight_quant.shape[0]
    rows = selected_rows(m, tokens_per_rank, chunk_tokens_per_rank, chunk_token_offset)
    if k_gamma.shape != (256,):
        raise ValueError("The current operator requires one-dimensional gamma=(256,)")
    if m and positions_contiguous and cos_sin.shape[0] < m:
        raise ValueError("Continuous test positions require cos_sin.shape[0] >= M")
    generator = torch.Generator().manual_seed(42)
    hidden = torch.randn((m, k), generator=generator).bfloat16().float()
    weight = (torch.randn((n, k), generator=generator) * 0.02).bfloat16().float()
    quantize_mx(hidden, hidden_quant, hidden_scale)
    quantize_mx(weight, weight_quant, weight_scale)
    k_gamma.copy_(torch.linspace(0.75, 1.25, 256).bfloat16())
    positions.zero_()
    p = cos_sin.shape[0]
    if positions_contiguous:
        positions[rows] = torch.arange(m)
    else:
        positions[rows] = (torch.arange(m) * 7 + 3) % p
    angles = (
        torch.arange(p).float()[:, None]
        / (10000 ** (torch.arange(32).float() / 32))[None, :]
    )
    cos_sin.copy_(torch.cat((angles.cos(), angles.sin()), dim=1))
    slot_mapping.fill_(-1)
    count = min(m, k_cache.shape[0])
    slot_mapping[rows[:count]] = torch.randperm(k_cache.shape[0], generator=generator)[
        :count
    ]
    for value, target in enumerate(
        (q_out, k_out, v_out, k_cache, v_cache, mirror_k_out, mirror_v_out), start=7
    ):
        if target is not None:
            target.fill_(value)


def dequantize_mx(payload, scale):
    """Decode independent of the quantizer and of the production implementation."""
    exponent = scale.view(torch.uint8).reshape(payload.shape[0], payload.shape[1] // 32)
    factors = torch.pow(2.0, exponent.double() - 127)
    return payload.double() * factors.repeat_interleave(32, dim=1)


def golden_all(
    hidden_quant,
    hidden_scale,
    weight_quant,
    weight_scale,
    k_gamma,
    positions,
    cos_sin,
    slot_mapping,
    q_out,
    k_out,
    v_out,
    k_cache,
    v_cache,
    *,
    epsilon=1e-6,
    mirror_k_out=None,
    mirror_v_out=None,
    positions_contiguous=False,
    tokens_per_rank=None,
    chunk_tokens_per_rank=None,
    chunk_token_offset=None,
    **kwargs,
):
    """Return five/seven CPU results without mutating any caller input.

    Follow the operator contract: FP64 dequant/projection -> BF16; K-only FP32
    RMSNorm -> BF16; NeoX RoPE on the final 64 features -> BF16; cache scatter.
    Mirrors are raw projection slices, without norm or RoPE.
    """
    has_mirror = mirror_k_out is not None
    if has_mirror != (mirror_v_out is not None):
        raise ValueError("mirror outputs must both be present or both absent")
    m = hidden_quant.shape[0]
    rows = selected_rows(m, tokens_per_rank, chunk_tokens_per_rank, chunk_token_offset)
    result = [tensor.clone() for tensor in (q_out, k_out, v_out, k_cache, v_cache)]
    if has_mirror:
        result.extend((mirror_k_out.clone(), mirror_v_out.clone()))
    if m == 0:
        return result
    projected = (
        dequantize_mx(hidden_quant, hidden_scale)
        @ dequantize_mx(weight_quant, weight_scale).T
    ).bfloat16()
    qw, kvw = q_out.shape[1], k_out.shape[1]
    key = projected[:, qw : qw + kvw].reshape(m, -1, 256).float()
    key = (
        key
        * torch.rsqrt(key.square().mean(-1, keepdim=True) + epsilon)
        * k_gamma.float()
    ).bfloat16()
    table = cos_sin[positions[rows]]

    def rope(values):
        values = values.reshape(m, -1, 256)
        output = values.clone()
        left, right = values[..., 192:224].float(), values[..., 224:256].float()
        cosine, sine = table[:, None, :32], table[:, None, 32:]
        output[..., 192:224] = (left * cosine - right * sine).bfloat16()
        output[..., 224:256] = (left * sine + right * cosine).bfloat16()
        return output.reshape(m, -1)

    q, key_out, value, kc, vc = result[:5]
    q[rows] = rope(projected[:, :qw])
    key_out[rows] = rope(key)
    value[rows] = projected[:, qw + kvw : qw + 2 * kvw]
    slots = slot_mapping[rows]
    valid = slots >= 0
    kc[slots[valid]] = key_out[rows[valid]]
    vc[slots[valid]] = value[rows[valid]]
    if has_mirror:
        result[5][rows] = projected[:, qw + 2 * kvw : qw + 3 * kvw]
        result[6][rows] = projected[:, qw + 3 * kvw : qw + 4 * kvw]
    return result


class FusedQuantQkvProjNormRopeCacheGraph(torch.nn.Module):
    """Expose five/seven mutable buffers to TTK's compiled graph comparison."""

    def forward(
        self,
        hidden_quant,
        hidden_scale,
        weight_quant,
        weight_scale,
        k_gamma,
        positions,
        cos_sin,
        slot_mapping,
        q_out,
        k_out,
        v_out,
        k_cache,
        v_cache,
        *,
        epsilon=1e-6,
        mirror_k_out=None,
        mirror_v_out=None,
        positions_contiguous=False,
        tokens_per_rank=None,
        chunk_tokens_per_rank=None,
        chunk_token_offset=None,
    ):
        # Keep scale encodings unchanged while presenting byte tensors to
        # functionalization; the opaque operator restores E8M0 for the provider.
        torch.ops.cann_ops_transformer.fused_quant_qkv_proj_norm_rope_cache(
            hidden_quant,
            hidden_scale.view(torch.int8),
            weight_quant,
            weight_scale.view(torch.int8),
            k_gamma,
            positions,
            cos_sin,
            slot_mapping,
            q_out,
            k_out,
            v_out,
            k_cache,
            v_cache,
            epsilon=epsilon,
            mirror_k_out=mirror_k_out,
            mirror_v_out=mirror_v_out,
            positions_contiguous=positions_contiguous,
            tokens_per_rank=tokens_per_rank,
            chunk_tokens_per_rank=chunk_tokens_per_rank,
            chunk_token_offset=chunk_token_offset,
        )
        outputs = (q_out, k_out, v_out, k_cache, v_cache)
        if mirror_k_out is not None:
            outputs += (mirror_k_out, mirror_v_out)
        return outputs


def _bf16_ordered_codes(values):
    # Collapse signed zero and order the negative BF16 codes by numerical value.
    signed = values.view(np.int16).astype(np.int32)
    return np.where(signed < 0, -32768 - signed, signed)


def compare_bf16_outputs(*outputs_and_goldens):
    """Compare BF16 outputs with dual 1e-3 tolerances or one BF16 step."""
    output_counts = {10: (5, 5), 12: (5, 7), 14: (7, 7)}
    counts = output_counts.get(len(outputs_and_goldens))
    if counts is None:
        raise ValueError(
            "expected 5 actual + 5/7 golden or 7 actual + 7 golden outputs"
        )
    actual_count, golden_count = counts
    actual_outputs = outputs_and_goldens[:actual_count]
    golden_outputs = outputs_and_goldens[actual_count:]
    if len(golden_outputs) != golden_count:
        raise ValueError("unexpected number of golden outputs")

    results = []
    for actual, golden in zip(actual_outputs, golden_outputs[:actual_count]):
        if actual is None and golden is None:
            results.append({"pass": True, "precision": 100.0})
            continue
        actual, golden = np.asarray(actual), np.asarray(golden)
        if (
            actual.shape != golden.shape
            or actual.dtype.name != "bfloat16"
            or golden.dtype != actual.dtype
        ):
            results.append(
                {
                    "pass": False,
                    "precision": 0.0,
                    "error_info": (
                        f"expected matching BF16 outputs, got {actual.shape}/{actual.dtype} "
                        f"and {golden.shape}/{golden.dtype}"
                    ),
                }
            )
            continue

        actual, golden = actual.reshape(-1), golden.reshape(-1)
        total = actual.size
        bad = 0
        diff_indices = []
        chunk_size = 1 << 20
        for begin in range(0, total, chunk_size):
            actual_chunk = actual[begin : begin + chunk_size]
            golden_chunk = golden[begin : begin + chunk_size]
            if np.array_equal(actual_chunk, golden_chunk):
                continue

            a, g = actual_chunk.astype(np.float64), golden_chunk.astype(np.float64)
            finite = np.isfinite(a) & np.isfinite(g)
            with np.errstate(invalid="ignore", over="ignore"):
                close = np.abs(a - g) <= 1e-3 + 1e-3 * np.abs(g)
            within_one_ulp = (
                np.abs(
                    _bf16_ordered_codes(actual_chunk)
                    - _bf16_ordered_codes(golden_chunk)
                )
                <= 1
            )
            # Match TTK close for NaNs and infinities; only finite values have ULP distance.
            same_value = (a == g) | (np.isnan(a) & np.isnan(g))
            mismatches = ~(same_value | (finite & (close | within_one_ulp)))
            bad += int(np.count_nonzero(mismatches))
            if len(diff_indices) < 20:
                first = np.flatnonzero(mismatches)[: 20 - len(diff_indices)]
                diff_indices.extend((first + begin).tolist())

        passed = bad * 1000 <= total
        result = {
            "pass": passed,
            "precision": 100.0 * (total - bad) / total if total else 100.0,
            "diff_indices": diff_indices,
            "metrics": {"mismatch_count": bad, "element_count": total},
        }
        if not passed:
            result["error_info"] = (
                f"{bad}/{total} elements exceed both tolerances (ptol=0.001)"
            )
        results.append(result)
    return results


class FusedQuantQkvProjNormRopeCacheE2eTestSpec:
    customize_inputs = staticmethod(customize_inputs)
    torch_graph = FusedQuantQkvProjNormRopeCacheGraph
    compare = staticmethod(compare_bf16_outputs)

    @staticmethod
    def golden(*args, **kwargs):
        return golden_all(*args, **kwargs)
