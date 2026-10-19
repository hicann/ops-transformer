#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""CPU golden and binary inputs for MoeTokenUnpermuteGrad kernel UTs."""

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


def get_cpu_data(
    token_num,
    topk,
    hidden_size,
    permuted_tokens,
    sorted_indices,
    output_grad,
    probs,
    flag,
):
    """sorted_indices maps original token/top-k slots to permuted rows."""
    indices = sorted_indices.to(torch.int64)
    if not flag and topk != 1:
        raise ValueError("topk must be 1 when probs is absent")
    expanded_grad = (
        output_grad.float().unsqueeze(1).expand(token_num, topk, hidden_size)
    )
    probs_grad = None
    if flag:
        original_tokens = (
            permuted_tokens.float()
            .index_select(0, indices)
            .reshape(token_num, topk, hidden_size)
        )
        probs_grad = (expanded_grad * original_tokens).sum(-1).to(probs.dtype)
        expanded_grad = expanded_grad * probs.float().unsqueeze(-1)
    token_grad = torch.zeros(token_num * topk, hidden_size, dtype=torch.float32)
    token_grad.index_copy_(0, indices, expanded_grad.reshape(-1, hidden_size))
    return token_grad.to(permuted_tokens.dtype), probs_grad


def gen_data_and_golden(
    token_num, topk, hiddensize, tokendtype, probdtype, flag, output_dir="."
):
    if min(token_num, topk, hiddensize) <= 0:
        raise ValueError("token_num, topk and hiddensize must be positive")
    if not flag and topk != 1:
        raise ValueError("topk must be 1 when probs is absent")
    generator = torch.Generator().manual_seed(0)
    tokens = (
        torch.randint(
            -8, 9, (token_num * topk, hiddensize), generator=generator
        ).float()
        / 8
    ).to(DTYPES[tokendtype])
    indices = torch.randperm(token_num * topk, generator=generator).to(torch.int32)
    output_grad = (
        torch.randint(-8, 9, (token_num, hiddensize), generator=generator).float() / 8
    ).to(DTYPES[tokendtype])
    probs = None
    if flag:
        probs = (
            torch.randint(-8, 9, (token_num, topk), generator=generator).float() / 8
        ).to(DTYPES[probdtype])
    token_grad, probs_grad = get_cpu_data(
        token_num, topk, hiddensize, tokens, indices, output_grad, probs, flag
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_tensor(output_dir / "input_permuted_tokens.bin", tokens)
    write_tensor(output_dir / "input_sorted_indices.bin", indices)
    write_tensor(output_dir / "input_unpermuted_output_d.bin", output_grad)
    write_tensor(output_dir / "output_permuted_tokens_grad.bin", token_grad)
    if flag:
        write_tensor(output_dir / "input_probs.bin", probs)
        write_tensor(output_dir / "output_probs_grad.bin", probs_grad)
    else:
        for name in ("input_probs.bin", "output_probs_grad.bin"):
            (output_dir / name).unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("token_num", type=int)
    parser.add_argument("topk", type=int)
    parser.add_argument("hiddensize", type=int)
    parser.add_argument("tokendtype", choices=DTYPES)
    parser.add_argument("probdtype", choices=DTYPES)
    parser.add_argument("flag", type=parse_bool)
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args()
    gen_data_and_golden(**vars(args))
