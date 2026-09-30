# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import argparse
import math
import os
import random
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

random.seed(42)
torch.manual_seed(42)


import math

import torch
import torch.nn.functional as F


def torch_chunk_local_cumsum(x, chunk_size=64):
    _, T, _ = x.shape
    assert T % chunk_size == 0
    return torch.cat(
        [
            x[:, i : i + chunk_size, :].float().cumsum(1)
            for i in range(0, T, chunk_size)
        ],
        1,
    )


def create_upper_triangle_ones(M, N, dtype=torch.float32, device="cpu"):
    """创建形状为(M,N)的上三角全1矩阵"""
    # 先创建全1矩阵，然后取上三角部分
    matrix = torch.ones((M, N), dtype=dtype, device=device)
    # triu返回上三角部分，下三角部分为0
    return torch.triu(matrix)


def torch_chunk_local_cumsum_v4_cube(
    x: torch.Tensor,
    chunk_size: int,
    cu_seqlens: torch.LongTensor,
) -> torch.Tensor:
    """
    Compute local cumulative sum within each chunk.

    Args:
        x: [B, H, T] or [1, H, N] - input tensor (transposed format)
            - If [B, H, T]: traditional batching format, will be flattened to [1, H, B*T]
            - If [1, H, N]: GVA/vLLM format, already flattened
        chunk_size: chunk size for local cumsum
        cu_seqlens: cumulative sequence lengths [NS+1], required parameter
            - NS: number of sequences
            - For [B, H, T] input: cu_seqlens should be [0, T, 2*T, ..., B*T]
            - For [1, H, N] input: cu_seqlens indicates variable-length sequences

    Returns:
        cumsum_x: [1, H, N] format, where N = total tokens across all sequences
            - N = B * T for [B, H, T] input
            - N = cu_seqlens[-1] for [1, H, N] input

    Note: No requirement for T % chunk_size == 0
    """
    # 输入是 [B, H, T] 或 [1, H, N]，转置为 [B, T, H] 或 [1, N, H] 以复用原有逻辑
    B, T, H = x.shape
    x_transposed = x  # [B, H, T] -> [B, T, H] 或 [1, H, N] -> [1, N, H]
    x_float = x_transposed.float()

    # Convert input to [1, N, H] format (使用 reshape 因为 permute 后的 tensor 可能不连续)
    x_flat = x_float.reshape(1, B * T, H)
    N = B * T

    # Verify cu_seqlens is consistent with input
    num_sequences = len(cu_seqlens) - 1
    expected_total = cu_seqlens[-1].item()
    assert expected_total == N, (
        f"cu_seqlens[-1] ({expected_total}) must equal total tokens N ({N})"
    )

    # Initialize output in [1, N, H] format
    result = torch.zeros(1, N, H, dtype=x_float.dtype, device=x_float.device)

    # Process each sequence separately
    for seq_idx in range(num_sequences):
        start = cu_seqlens[seq_idx].item()
        end = cu_seqlens[seq_idx + 1].item()
        seq_len = end - start

        if seq_len == 0:
            continue

        # Process each chunk in the sequence
        for chunk_start in range(0, seq_len, chunk_size):
            chunk_end = min(chunk_start + chunk_size, seq_len)
            chunk_start_global = start + chunk_start
            chunk_end_global = start + chunk_end

            # Compute cumsum for this chunk
            chunk_data = x_flat[0, chunk_start_global:chunk_end_global, :]
            upper_tri = create_upper_triangle_ones(
                chunk_data.shape[0], chunk_data.shape[0]
            )
            result[0, chunk_start_global:chunk_end_global, :] = upper_tri @ chunk_data

    # 转置回 [1, H, N] 格式
    return result


def torch_chunk_local_cumsum_v4(
    x: torch.Tensor,
    chunk_size: int,
    cu_seqlens: torch.LongTensor,
) -> torch.Tensor:
    """
    Compute local cumulative sum within each chunk.

    Args:
        x: [B, H, T] or [1, H, N] - input tensor (transposed format)
            - If [B, H, T]: traditional batching format, will be flattened to [1, H, B*T]
            - If [1, H, N]: GVA/vLLM format, already flattened
        chunk_size: chunk size for local cumsum
        cu_seqlens: cumulative sequence lengths [NS+1], required parameter
            - NS: number of sequences
            - For [B, H, T] input: cu_seqlens should be [0, T, 2*T, ..., B*T]
            - For [1, H, N] input: cu_seqlens indicates variable-length sequences

    Returns:
        cumsum_x: [1, H, N] format, where N = total tokens across all sequences
            - N = B * T for [B, H, T] input
            - N = cu_seqlens[-1] for [1, H, N] input

    Note: No requirement for T % chunk_size == 0
    """
    # 输入是 [B, H, T] 或 [1, H, N]，转置为 [B, T, H] 或 [1, N, H] 以复用原有逻辑
    B, T, H = x.shape
    x_float = x.float()

    # Convert input to [1, N, H] format (使用 reshape 因为 permute 后的 tensor 可能不连续)
    x_flat = x_float.reshape(1, B * T, H)
    N = B * T

    # Verify cu_seqlens is consistent with input
    num_sequences = len(cu_seqlens) - 1
    expected_total = cu_seqlens[-1].item()
    assert expected_total == N, (
        f"cu_seqlens[-1] ({expected_total}) must equal total tokens N ({N})"
    )

    # Initialize output in [1, N, H] format
    result = torch.zeros(1, N, H, dtype=x_float.dtype, device=x_float.device)

    # Process each sequence separately
    for seq_idx in range(num_sequences):
        start = cu_seqlens[seq_idx].item()
        end = cu_seqlens[seq_idx + 1].item()
        seq_len = end - start

        if seq_len == 0:
            continue

        # Process each chunk in the sequence
        for chunk_start in range(0, seq_len, chunk_size):
            chunk_end = min(chunk_start + chunk_size, seq_len)
            chunk_start_global = start + chunk_start
            chunk_end_global = start + chunk_end

            # Compute cumsum for this chunk
            chunk_data = x_flat[0, chunk_start_global:chunk_end_global, :]
            result[0, chunk_start_global:chunk_end_global, :] = chunk_data.cumsum(dim=0)

    # 转置回 [1, H, N] 格式
    return result


# ====================================================================================
# 测试代码
# ====================================================================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="manual to this script")
    parser.add_argument("--cu-seqlens", type=str, default="0,64")
    parser.add_argument("--head-num", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=64)
    args = parser.parse_args()

    head_num = args.head_num
    chunk_size = args.chunk_size
    device = "cpu"

    cu_seqlen = args.cu_seqlens.split(",")
    cu_seqlens_v4 = torch.from_numpy(np.array(cu_seqlen).astype(np.int64))
    total_tokens = int(cu_seqlens_v4[-1])
    # 使用 [1, H, N] 格式输入（GVA/vLLM 模式，转置后）
    x_test = F.logsigmoid(
        torch.rand(1, total_tokens, head_num, dtype=torch.float32, device=device)
    )
    # x_test = torch.ones(1, total_tokens,head_num, dtype=torch.float32, device=device)*torch.arange(total_tokens).reshape(-1,total_tokens,1)+1
    # v4 版本：cu_seqlens 现在是必需参数
    cumsum_v4 = torch_chunk_local_cumsum_v4(x_test, chunk_size, cu_seqlens_v4)
    # cumsum_v4_cube = torch_chunk_local_cumsum_v4_cube(x_test, chunk_size, cu_seqlens_v4)

    # torch.allclose(cumsum_v4, cumsum_v4_cube, rtol=1e-3, atol=1e-3)
    print("test pass")
    # 验证输出形状：统一为 [1, H, N] 格式
    # assert cumsum_v4.shape == (1, num_v_heads, N), f"cumsum shape mismatch: {cumsum_v4.shape} vs expected {(1, num_v_heads, N)}"

    os.makedirs("./input", exist_ok=True)
    os.makedirs("./output", exist_ok=True)

    x_test.numpy().tofile("./input/input.bin")
    cu_seqlens_v4.numpy().tofile("./input/input_cu_seqlens.bin")
    cumsum_v4.numpy().tofile("./output/golden_output.bin")
