#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""CPU golden and binary inputs for MoeTokenPermuteWithEpGrad kernel UTs."""

import argparse
from pathlib import Path

import torch


DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bfloat16_t": torch.bfloat16,
}


def parse_bool(value):
    if value not in ("True", "False"):
        raise argparse.ArgumentTypeError("expected True or False")
    return value == "True"


def write_tensor(path, tensor):
    tensor = tensor.detach().cpu().contiguous()
    # Preserve BF16 storage without requiring a NumPy BF16 extension.
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.view(torch.int16)
    tensor.numpy().tofile(path)


def permute_with_ep_grad(
    permuted_tokens, sorted_indices, top_k=1, probs=None, start=0, end=None
):
    """sorted_indices maps each original token/top-k slot to a permuted row."""
    total = sorted_indices.numel()
    end = total if end is None else end
    if not 0 <= start < end <= total:
        raise ValueError("expected 0 <= start < end <= tokens_num * topk")
    if permuted_tokens.shape[0] != end - start:
        raise ValueError("permuted token rows must equal end - start")
    # Accumulate in original top-k order, matching the kernel's FP32 addition order.
    indices = sorted_indices.to(torch.int64)
    valid = (indices >= start) & (indices < end)
    gathered = permuted_tokens.float()[torch.clamp(indices - start, 0, end - start - 1)]
    gathered[~valid] = 0
    gathered = gathered.reshape(total // top_k, top_k, -1)
    tokens_out = gathered[:, 0].clone()
    for k in range(1, top_k):
        tokens_out += gathered[:, k]
    probs_out = None
    if probs is not None:
        probs_out = probs.reshape(-1)[
            torch.clamp(indices - start, 0, end - start - 1)
        ].clone()
        probs_out[~valid] = 0
        probs_out = probs_out.reshape(-1, top_k)
    return tokens_out.to(permuted_tokens.dtype), probs_out


def gen_data_and_golden(
    tokens_num,
    topk,
    hidden_size,
    tokendtype,
    have_probs,
    output_dir=".",
    start=0,
    end=None,
):
    if min(tokens_num, topk, hidden_size) <= 0:
        raise ValueError("tokens_num, topk and hidden_size must be positive")
    total = tokens_num * topk
    end = total if end is None else end
    if not 0 <= start < end <= total:
        raise ValueError("expected 0 <= start < end <= tokens_num * topk")
    generator = torch.Generator().manual_seed(0)
    dtype = DTYPES[tokendtype]
    # Binary fractions make the small kernel UTs independent of reduction rounding.
    tokens = (
        torch.randint(-8, 9, (end - start, hidden_size), generator=generator).float()
        / 8
    ).to(dtype)
    indices = torch.randperm(total, generator=generator).to(torch.int32)
    probs = None
    if have_probs:
        probs = (
            torch.randint(-8, 9, (end - start,), generator=generator).float() / 8
        ).to(dtype)
    token_grad, probs_grad = permute_with_ep_grad(
        tokens, indices, topk, probs, start, end
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_tensor(output_dir / "input_permuted_tokens.bin", tokens)
    write_tensor(output_dir / "input_sorted_indices.bin", indices)
    write_tensor(output_dir / "output_unpermuted_tokens.bin", token_grad)
    if have_probs:
        write_tensor(output_dir / "input_probs.bin", probs)
        write_tensor(output_dir / "output_probs_grad.bin", probs_grad)
    else:
        for name in ("input_probs.bin", "output_probs_grad.bin"):
            (output_dir / name).unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tokens_num", type=int)
    parser.add_argument("topk", type=int)
    parser.add_argument("hidden_size", type=int)
    parser.add_argument("tokendtype", choices=DTYPES)
    parser.add_argument("have_probs", type=parse_bool)
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int)
    args = parser.parse_args()
    gen_data_and_golden(**vars(args))
