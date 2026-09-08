#!/usr/bin/python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

import os
import sys

import numpy as np
import torch

# Precision standard for MXFP8 quantized matmul with BF16 output
# Reference: DESIGN.md § 精度分析
ERROR_TOL_RTOL = 0.02  # 2% relative tolerance
ERROR_TOL_ATOL = 0.2  # absolute tolerance
DATA_TYPE = np.uint16
FULL_TENSOR_PRINT_MAX_ELEMENTS = 1024
CORNER_ROWS = 4
CORNER_COLS = 4


def _print_large_tensor_summary(
    golden_tensor: torch.Tensor,
    npu_output_tensor: torch.Tensor,
    m_total: int,
    n_per_rank: int,
    rank_id: int = -1,
) -> None:
    """Print summary statistics for large tensors."""
    g = golden_tensor.float()
    p = npu_output_tensor.float()
    diff = p - g
    abs_err = diff.abs()
    denom = g.abs().clamp_min(1e-8)
    rel_err = abs_err / denom

    numel = m_total * n_per_rank
    over_tol = (abs_err > ERROR_TOL_ATOL).sum().item()

    rank_prefix = f"[Rank {rank_id}] " if rank_id >= 0 else ""
    print(
        f"\n{rank_prefix}[verify] shape=({m_total}, {n_per_rank}), elements={numel} - "
        f"summary (large matrix, full tensors omitted)"
    )
    print(
        f"  abs_err: max={abs_err.max().item():.6e}, mean={abs_err.mean().item():.6e}, "
        f"rmse={(diff.pow(2).mean().sqrt()).item():.6e}"
    )
    print(f"  rel_err: max={rel_err.max().item():.6e}")
    print(f"  count(|abs_err| > {ERROR_TOL_ATOL:g}): {over_tol} / {numel}")

    cr = min(CORNER_ROWS, m_total)
    cc = min(CORNER_COLS, n_per_rank)
    if cr > 0 and cc > 0:
        print(f"  cpu golden (top-left {cr}x{cc}):\n{golden_tensor[:cr, :cc]}")
        print(f"  npu output (top-left {cr}x{cc}):\n{npu_output_tensor[:cr, :cc]}")


def verify_single_rank(
    m_total: int,
    n_per_rank: int,
    rank_id: int,
    base_dir: str = "./output",
    out_dtype: torch.dtype = torch.bfloat16,
) -> bool:
    """
    Verify NPU output for a single rank.

    Args:
        m_total: Total rows = rank_num * m (output shape)
        n_per_rank: Columns per rank = n / rank_num (output shape)
        rank_id: Rank ID to verify
        base_dir: Output directory base path
        out_dtype: Output dtype (torch.bfloat16 or torch.float16)

    Returns:
        bool: True if verification passes
    """
    output_path = os.path.join(base_dir, str(rank_id), "npu_out.bin")
    golden_path = os.path.join(base_dir, str(rank_id), "cpu_output.bin")

    if not os.path.exists(output_path):
        raise FileNotFoundError(f"NPU output file not found: {output_path}")
    if not os.path.exists(golden_path):
        raise FileNotFoundError(f"CPU golden file not found: {golden_path}")

    output = np.fromfile(output_path, dtype=DATA_TYPE)
    golden = np.fromfile(golden_path, dtype=DATA_TYPE)

    expected_size = m_total * n_per_rank
    if output.size != expected_size:
        raise ValueError(
            f"[Rank {rank_id}] npu output size {output.size} != expected size {expected_size}"
        )
    if golden.size != expected_size:
        raise ValueError(
            f"[Rank {rank_id}] cpu output size {golden.size} != expected size {expected_size}"
        )

    npu_output_tensor = (
        torch.from_numpy(output).view(out_dtype).reshape(m_total, n_per_rank)
    )
    golden_tensor = (
        torch.from_numpy(golden).view(out_dtype).reshape(m_total, n_per_rank)
    )

    numel = m_total * n_per_rank
    if numel <= FULL_TENSOR_PRINT_MAX_ELEMENTS:
        print(f"\n[Rank {rank_id}] cpu golden:\n", golden_tensor)
        print(f"[Rank {rank_id}] npu output:\n", npu_output_tensor)
    else:
        _print_large_tensor_summary(
            golden_tensor, npu_output_tensor, m_total, n_per_rank, rank_id
        )

    return torch.allclose(
        golden_tensor,
        npu_output_tensor,
        rtol=ERROR_TOL_RTOL,
        atol=ERROR_TOL_ATOL,
    )


def verify_result(
    m: int,
    n: int,
    rank_num: int,
    base_dir: str = "./output",
    out_dtype: torch.dtype = torch.bfloat16,
) -> bool:
    """
    Verify results for all ranks.

    Args:
        m: Matrix M dimension (per rank)
        n: Matrix N dimension (total)
        rank_num: Number of ranks
        base_dir: Output directory base path
        out_dtype: Output dtype (torch.bfloat16 or torch.float16)

    Returns:
        bool: True if all ranks pass verification
    """
    all_pass = True
    results = []

    # Calculate output dimensions
    m_total = rank_num * m  # Total rows in output
    n_per_rank = n // rank_num  # Columns per rank

    print(f"\n{'=' * 60}")
    print(f"Verifying outputs for {rank_num} ranks")
    print(f"Input matrix shape: M={m}, N={n}")
    print(f"Output matrix shape: ({m_total}, {n_per_rank})")
    print(f"Output dtype: {out_dtype}")
    print(f"Precision standard: rtol={ERROR_TOL_RTOL * 100}%, atol={ERROR_TOL_ATOL}")
    print(f"{'=' * 60}")

    for rank_id in range(rank_num):
        try:
            res = verify_single_rank(m_total, n_per_rank, rank_id, base_dir, out_dtype)
            if res:
                results.append((rank_id, True, None))
                print(f"[PASS] Rank {rank_id}: NPU results match CPU golden")
            else:
                results.append((rank_id, False, "precision mismatch"))
                print(f"[FAIL] Rank {rank_id}: NPU results differ from CPU golden")
                all_pass = False
        except Exception as e:
            results.append((rank_id, False, str(e)))
            print(f"[FAIL] Rank {rank_id}: {e}")
            all_pass = False

    print(f"\n{'=' * 60}")
    print("Verification Summary:")
    print(f"{'=' * 60}")
    for rank_id, passed, error in results:
        status = "PASS" if passed else "FAIL"
        print(f"  Rank {rank_id}: {status}")
        if error:
            print(f"    Error: {error}")

    return all_pass


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python3 verify_result.py m n rank_num [base_dir] [is_fp16]")
        print("  m: matrix M dimension (per rank)")
        print("  n: matrix N dimension (total)")
        print("  rank_num: number of ranks")
        print("  base_dir: output directory (default: ./output)")
        print("  is_fp16: output is float16 instead of bfloat16 (default: 0)")
        print("\nExpected output shape per rank:")
        print("  ({rank_num}*{m}, {n}/{rank_num})")
        print("\nExpected file structure:")
        print("  {base_dir}/0/npu_out.bin      - NPU output from rank 0")
        print("  {base_dir}/0/cpu_output.bin   - CPU golden for rank 0")
        print("  {base_dir}/1/npu_out.bin      - NPU output from rank 1")
        print("  {base_dir}/1/cpu_output.bin   - CPU golden for rank 1")
        print("  ...")
        print("\nExample: python3 verify_result.py 2048 4096 4 ./output 0")
        sys.exit(1)

    m = int(sys.argv[1])
    n = int(sys.argv[2])
    rank_num = int(sys.argv[3])
    base_dir = sys.argv[4] if len(sys.argv) > 4 else "./output"
    is_fp16 = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    out_dtype = torch.float16 if is_fp16 else torch.bfloat16

    # Validate parameters
    if rank_num <= 0:
        print(f"Error: rank_num={rank_num} must be a positive integer")
        sys.exit(1)
    if n % rank_num != 0:
        print(f"Error: n={n} is not divisible by rank_num={rank_num}")
        sys.exit(1)

    try:
        all_pass = verify_result(m, n, rank_num, base_dir, out_dtype)
        if not all_pass:
            raise ValueError("[ERROR] Some NPU results differ from CPU.\n")
        print(f"\n[ALL PASS] All {rank_num} ranks passed verification!\n")

    except Exception as e:
        print(e)
        sys.exit(1)
