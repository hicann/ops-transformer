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

"""Executable NumPy golden for the HiF8 BlockSparseAttentionGrad path.

The HiF8 codec is loaded from fp32_hifp8_quant.py.  This program models the
kernel's exact scale algebra and supports both static and per-sparse-block dS
quantization.  It does not require torch or an NPU.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np


def load_codec(path: Path):
    spec = importlib.util.spec_from_file_location("fp32_hifp8_quant", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load HiF8 codec: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bf16_rne(x: np.ndarray) -> np.ndarray:
    """Round FP32 to BF16 (RNE) and return the representable values as FP32."""
    value = np.asarray(x, dtype=np.float32).copy()
    bits = value.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)
    return (rounded & np.uint32(0xFFFF0000)).view(np.float32)


def bf16_bits(x: np.ndarray) -> np.ndarray:
    """Return little-endian BF16 payloads for an FP32 array."""
    rounded = bf16_rne(x).view(np.uint32)
    return (rounded >> 16).astype("<u2")


def quant(
    codec: Any, x: np.ndarray, descale: float | None = None
) -> tuple[np.ndarray, np.ndarray, float]:
    if descale is None:
        code, scale = codec.quantize(x, mode="per_tensor")
    else:
        scale = np.asarray(descale, dtype=np.float32)
        code, _ = codec.quantize(x, scale=scale)
    encoded = codec.hifp8_to_fp32(code).astype(np.float32)
    return code, encoded, float(scale)


def make_mask(heads: int, q_blocks: int, kv_blocks: int) -> np.ndarray:
    """A deterministic irregular pattern; every query block has a valid KV block."""
    mask = np.zeros((heads, q_blocks, kv_blocks), dtype=bool)
    for h in range(heads):
        for qb in range(q_blocks):
            mask[h, qb, (qb + h) % kv_blocks] = True
            mask[h, qb, (2 * qb + h + 1) % kv_blocks] = True
    return mask


def forward(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    mask: np.ndarray,
    block_m: int,
    block_n: int,
    scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    batch, q_heads, sq, _ = q.shape
    kv_heads, skv = k.shape[1], k.shape[2]
    group = q_heads // kv_heads
    probability = np.zeros((batch, q_heads, sq, skv), dtype=np.float32)
    lse = np.full((batch, q_heads, sq), -np.inf, dtype=np.float32)
    out = np.zeros_like(q, dtype=np.float32)
    for b in range(batch):
        for h in range(q_heads):
            kh = h // group
            for qb in range((sq + block_m - 1) // block_m):
                qs = slice(qb * block_m, min((qb + 1) * block_m, sq))
                cols: list[int] = []
                for kb in range((skv + block_n - 1) // block_n):
                    if mask[h, qb, kb]:
                        cols.extend(range(kb * block_n, min((kb + 1) * block_n, skv)))
                col = np.asarray(cols, dtype=np.int64)
                score = (q[b, h, qs] @ k[b, kh, col].T) * np.float32(scale)
                row_max = np.max(score, axis=-1, keepdims=True)
                exp = np.exp(score - row_max, dtype=np.float32)
                denom = np.sum(exp, axis=-1, keepdims=True, dtype=np.float32)
                prob = exp / denom
                probability[b, h, qs][:, col] = prob
                lse[b, h, qs] = np.log(denom[:, 0]) + row_max[:, 0]
                out[b, h, qs] = prob @ v[b, kh, col]
    return bf16_rne(out), lse, probability


def backward_reference(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    dout: np.ndarray,
    out_bf16: np.ndarray,
    probability: np.ndarray,
    mask: np.ndarray,
    block_m: int,
    block_n: int,
    scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    batch, q_heads, sq, dim = q.shape
    kv_heads, skv = k.shape[1], k.shape[2]
    group = q_heads // kv_heads
    dq = np.zeros((batch, q_heads, sq, dim), dtype=np.float32)
    dk = np.zeros((batch, kv_heads, skv, dim), dtype=np.float32)
    dv = np.zeros_like(dk)
    delta = np.sum(dout * out_bf16, axis=-1, dtype=np.float32)
    for b in range(batch):
        for h in range(q_heads):
            kh = h // group
            for qb in range((sq + block_m - 1) // block_m):
                qs = slice(qb * block_m, min((qb + 1) * block_m, sq))
                for kb in range((skv + block_n - 1) // block_n):
                    if not mask[h, qb, kb]:
                        continue
                    ks = slice(kb * block_n, min((kb + 1) * block_n, skv))
                    p = probability[b, h, qs, ks]
                    dp = dout[b, h, qs] @ v[b, kh, ks].T
                    ds = (dp - delta[b, h, qs, None]) * p
                    dq[b, h, qs] += (ds @ k[b, kh, ks]) * np.float32(scale)
                    dk[b, kh, ks] += (ds.T @ q[b, h, qs]) * np.float32(scale)
                    dv[b, kh, ks] += p.T @ dout[b, h, qs]
    return dq, dk, dv


def backward_hif8(
    codec: Any,
    q_code: np.ndarray,
    k_code: np.ndarray,
    v_code: np.ndarray,
    do_code: np.ndarray,
    q_desc: float,
    k_desc: float,
    v_desc: float,
    do_desc: float,
    out_bf16: np.ndarray,
    probability: np.ndarray,
    mask: np.ndarray,
    block_m: int,
    block_n: int,
    scale: float,
    ds_mode: str,
) -> tuple[tuple[np.ndarray, ...], dict[str, float]]:
    q8 = codec.hifp8_to_fp32(q_code).astype(np.float32)
    k8 = codec.hifp8_to_fp32(k_code).astype(np.float32)
    v8 = codec.hifp8_to_fp32(v_code).astype(np.float32)
    do8 = codec.hifp8_to_fp32(do_code).astype(np.float32)
    dout = do8 * np.float32(do_desc)  # only the rowsum branch is explicitly dequantized
    batch, q_heads, sq, dim = q8.shape
    kv_heads, skv = k8.shape[1], k8.shape[2]
    group = q_heads // kv_heads
    p_scale = 32768.0
    dq = np.zeros((batch, q_heads, sq, dim), dtype=np.float32)
    dk = np.zeros((batch, kv_heads, skv, dim), dtype=np.float32)
    dv = np.zeros_like(dk)
    delta = np.sum(dout * out_bf16, axis=-1, dtype=np.float32)
    ds_blocks: list[np.ndarray] = []
    blocks: list[tuple[int, int, int, slice, slice, np.ndarray]] = []
    for b in range(batch):
        for h in range(q_heads):
            kh = h // group
            for qb in range((sq + block_m - 1) // block_m):
                qs = slice(qb * block_m, min((qb + 1) * block_m, sq))
                for kb in range((skv + block_n - 1) // block_n):
                    if not mask[h, qb, kb]:
                        continue
                    ks = slice(kb * block_n, min((kb + 1) * block_n, skv))
                    p_code, p_encoded, _ = quant(
                        codec, probability[b, h, qs, ks], 1.0 / p_scale
                    )
                    p_deq = p_encoded / np.float32(p_scale)
                    dp = (do8[b, h, qs] @ v8[b, kh, ks].T) * np.float32(
                        do_desc * v_desc
                    )
                    ds = (dp - delta[b, h, qs, None]) * p_deq
                    ds_blocks.append(ds)
                    blocks.append((b, h, kh, qs, ks, p_encoded))
    static_multiplier = 32768.0 / max(
        max(float(np.max(np.abs(x))) for x in ds_blocks), 1.0e-12
    )
    used_scales: list[float] = []
    for ds, (b, h, kh, qs, ks, p_encoded) in zip(ds_blocks, blocks):
        multiplier = (
            static_multiplier
            if ds_mode == "static"
            else 32768.0 / max(float(np.max(np.abs(ds))), 1.0e-12)
        )
        _, ds_encoded, _ = quant(codec, ds, 1.0 / multiplier)
        ds_desc = 1.0 / multiplier
        used_scales.append(ds_desc)
        dq[b, h, qs] += (ds_encoded @ k8[b, kh, ks]) * np.float32(
            ds_desc * k_desc * scale
        )
        dk[b, kh, ks] += (ds_encoded.T @ q8[b, h, qs]) * np.float32(
            ds_desc * q_desc * scale
        )
        dv[b, kh, ks] += (p_encoded.T @ do8[b, h, qs]) * np.float32(do_desc / p_scale)
    stats = {
        "p_scale_multiplier": p_scale,
        "static_ds_multiplier": static_multiplier,
        "min_ds_descale": min(used_scales),
        "max_ds_descale": max(used_scales),
        "sparse_tasks": float(len(blocks)),
    }
    return (bf16_rne(dq), bf16_rne(dk), bf16_rne(dv)), stats


def metrics(actual: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    a = actual.astype(np.float64).ravel()
    e = expected.astype(np.float64).ravel()
    diff = a - e
    denom = np.linalg.norm(a) * np.linalg.norm(e)
    return {
        "max_abs": float(np.max(np.abs(diff))),
        "mean_abs": float(np.mean(np.abs(diff))),
        "relative_l2": float(np.linalg.norm(diff) / max(np.linalg.norm(e), 1.0e-30)),
        "cosine": float(np.dot(a, e) / denom) if denom else 1.0,
    }


def save_raw_case(
    raw_dir: Path,
    mode: str,
    q_code: np.ndarray,
    k_code: np.ndarray,
    v_code: np.ndarray,
    do_code: np.ndarray,
    out_bf16: np.ndarray,
    lse: np.ndarray,
    mask: np.ndarray,
    descales: tuple[float, float, float, float],
    p_scale: float,
    ds_scale: float,
    quant_out: tuple[np.ndarray, ...],
    report: dict[str, Any],
) -> None:
    """Write a C++/AscendCL-friendly binary case plus its manifest."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        "q_hif8.bin": np.asarray(q_code, dtype=np.uint8),
        "k_hif8.bin": np.asarray(k_code, dtype=np.uint8),
        "v_hif8.bin": np.asarray(v_code, dtype=np.uint8),
        "dout_hif8.bin": np.asarray(do_code, dtype=np.uint8),
        "attention_out_bf16.bin": bf16_bits(out_bf16),
        "softmax_lse_fp32.bin": np.asarray(lse, dtype="<f4"),
        "block_sparse_mask_u8.bin": np.asarray(mask[None, ...], dtype=np.uint8),
        "q_descale_fp32.bin": np.asarray([descales[0]], dtype="<f4"),
        "k_descale_fp32.bin": np.asarray([descales[1]], dtype="<f4"),
        "v_descale_fp32.bin": np.asarray([descales[2]], dtype="<f4"),
        "dout_descale_fp32.bin": np.asarray([descales[3]], dtype="<f4"),
        "p_scale_fp32.bin": np.asarray([p_scale], dtype="<f4"),
        "ds_scale_fp32.bin": np.asarray([ds_scale], dtype="<f4"),
        "expected_dq_bf16.bin": bf16_bits(quant_out[0]),
        "expected_dk_bf16.bin": bf16_bits(quant_out[1]),
        "expected_dv_bf16.bin": bf16_bits(quant_out[2]),
    }
    for name, value in arrays.items():
        value.tofile(raw_dir / name)
    manifest = {
        "mode": mode,
        "shape": report["shape"],
        "block_shape": report["block_shape"],
        "files": {
            name: {"elements": int(value.size), "dtype": str(value.dtype)}
            for name, value in arrays.items()
        },
        "report": report,
    }
    (raw_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )


def run(
    codec: Any, mode: str, seed: int, save_npz: Path | None, save_raw_dir: Path | None
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    batch, q_heads, kv_heads, sq, skv, dim = 1, 4, 2, 128, 128, 128
    block_m = block_n = 32
    softmax_scale = dim**-0.5
    q = rng.normal(0, 0.35, (batch, q_heads, sq, dim)).astype(np.float32)
    k = rng.normal(0, 0.35, (batch, kv_heads, skv, dim)).astype(np.float32)
    v = rng.normal(0, 0.25, (batch, kv_heads, skv, dim)).astype(np.float32)
    dout = rng.normal(0, 0.2, (batch, q_heads, sq, dim)).astype(np.float32)
    q_code, _, q_desc = quant(codec, q)
    k_code, _, k_desc = quant(codec, k)
    v_code, _, v_desc = quant(codec, v)
    do_code, _, do_desc = quant(codec, dout)
    q_deq = codec.dequantize(q_code, q_desc)
    k_deq = codec.dequantize(k_code, k_desc)
    v_deq = codec.dequantize(v_code, v_desc)
    do_deq = codec.dequantize(do_code, do_desc)
    mask = make_mask(q_heads, sq // block_m, skv // block_n)
    out_bf16, lse, probability = forward(
        q_deq, k_deq, v_deq, mask, block_m, block_n, softmax_scale
    )
    fp32_ref = backward_reference(
        q_deq,
        k_deq,
        v_deq,
        do_deq,
        out_bf16,
        probability,
        mask,
        block_m,
        block_n,
        softmax_scale,
    )
    quant_out, scale_stats = backward_hif8(
        codec,
        q_code,
        k_code,
        v_code,
        do_code,
        q_desc,
        k_desc,
        v_desc,
        do_desc,
        out_bf16,
        probability,
        mask,
        block_m,
        block_n,
        softmax_scale,
        mode,
    )
    report: dict[str, Any] = {
        "mode": mode,
        "shape": {
            "B": batch,
            "Nq": q_heads,
            "Nkv": kv_heads,
            "Sq": sq,
            "Skv": skv,
            "D": dim,
        },
        "block_shape": [block_m, block_n],
        "input_descales": {"q": q_desc, "k": k_desc, "v": v_desc, "dout": do_desc},
        "scale_stats": scale_stats,
        "metrics_vs_dequantized_input_fp32": {
            name: metrics(got, ref)
            for name, got, ref in zip(("dq", "dk", "dv"), quant_out, fp32_ref)
        },
    }
    if save_npz is not None:
        save_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            save_npz,
            q_hif8=q_code,
            k_hif8=k_code,
            v_hif8=v_code,
            dout_hif8=do_code,
            attention_out_bf16_values=out_bf16,
            softmax_lse=lse,
            block_sparse_mask=mask.astype(np.uint8),
            q_descale=q_desc,
            k_descale=k_desc,
            v_descale=v_desc,
            dout_descale=do_desc,
            dq=quant_out[0],
            dk=quant_out[1],
            dv=quant_out[2],
        )
    if save_raw_dir is not None:
        ds_scale = scale_stats["static_ds_multiplier"] if mode == "static" else 0.0
        save_raw_case(
            save_raw_dir,
            mode,
            q_code,
            k_code,
            v_code,
            do_code,
            out_bf16,
            lse,
            mask,
            (q_desc, k_desc, v_desc, do_desc),
            scale_stats["p_scale_multiplier"],
            ds_scale,
            quant_out,
            report,
        )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--codec", type=Path, required=True, help="path to fp32_hifp8_quant.py"
    )
    parser.add_argument(
        "--ds-mode", choices=("static", "dynamic", "both"), default="both"
    )
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--save-npz", type=Path)
    parser.add_argument(
        "--save-raw-dir",
        type=Path,
        help="write raw tensors and manifest for the AscendCL example",
    )
    parser.add_argument(
        "--report-json", type=Path, help="also write the printed report as JSON"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    codec = load_codec(args.codec)
    modes = ("static", "dynamic") if args.ds_mode == "both" else (args.ds_mode,)
    result = {}
    for mode in modes:
        output = args.save_npz
        raw_dir = args.save_raw_dir
        if output is not None and len(modes) > 1:
            output = output.with_name(f"{output.stem}_{mode}{output.suffix}")
        if raw_dir is not None and len(modes) > 1:
            raw_dir = raw_dir / mode
        result[mode] = run(codec, mode, args.seed, output, raw_dir)
    serialized = json.dumps(result, indent=2, sort_keys=True)
    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
