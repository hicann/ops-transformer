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

import argparse
import os
import sys

import numpy as np
import torch

# 双千分之一精度标准（适用于所有模式）：
#   单元素相对误差超 ERROR_TOL(1e-2) 的元素占比 ≤ FAIL_RATE_LIMIT(1e-2) 算过。
#   量化（FP8/MXFP8）输入 + BF16 输出，理论相对误差 ~0.5-1%；非量化 fp16/bf16 更高精度，同门槛足够。
ERROR_TOL = 1e-2
FAIL_RATE_LIMIT = 1e-2
FULL_TENSOR_PRINT_MAX_ELEMENTS = 1024
CORNER_ROWS = 4
CORNER_COLS = 4


def read_matrix(path, dtype):
    """读取 fp16/bf16 原始 2 字节矩阵。bf16 无 numpy 类型，经 uint16 视图还原。"""
    if dtype == "bf16":
        raw = np.fromfile(path, dtype=np.uint16)
        return torch.from_numpy(raw).view(torch.bfloat16)
    return torch.from_numpy(np.fromfile(path, dtype=np.float16))


def _print_large_tensor_summary(
    golden_tensor: torch.Tensor,
    npu_output_tensor: torch.Tensor,
    m: int,
    n: int,
    rank_id: int = -1,
) -> None:
    g = golden_tensor.float()
    p = npu_output_tensor.float()
    diff = p - g
    abs_err = diff.abs()
    denom = g.abs().clamp_min(1e-8)
    rel_err = abs_err / denom

    numel = m * n
    # 非有限值（NaN/Inf）也计入超差统计，避免 NaN 静默漏统计
    over_tol = ((~torch.isfinite(abs_err)) | (abs_err > ERROR_TOL)).sum().item()

    rank_prefix = f"[Rank {rank_id}] " if rank_id >= 0 else ""
    print(
        f"\n{rank_prefix}[verify] shape=({m}, {n}), elements={numel} - summary (large matrix, full tensors omitted)"
    )
    print(
        f"  abs_err: max={abs_err.max().item():.6e}, mean={abs_err.mean().item():.6e}, "
        f"rmse={(diff.pow(2).mean().sqrt()).item():.6e}"
    )
    print(f"  rel_err: max={rel_err.max().item():.6e}")
    print(f"  count(|abs_err| > {ERROR_TOL:g}): {over_tol} / {numel}")

    cr = min(CORNER_ROWS, m)
    cc = min(CORNER_COLS, n)
    if cr > 0 and cc > 0:
        print(f"  cpu golden (top-left {cr}x{cc}):\n{golden_tensor[:cr, :cc]}")
        print(f"  npu output (top-left {cr}x{cc}):\n{npu_output_tensor[:cr, :cc]}")


def verify_single_rank(m, n, rank_id, base_dir="./output", dtype="bf16"):
    output_path = os.path.join(base_dir, str(rank_id), "npu_out.bin")
    golden_path = os.path.join(base_dir, str(rank_id), "cpu_output.bin")

    if not os.path.exists(output_path):
        raise FileNotFoundError(f"NPU output file not found: {output_path}")
    if not os.path.exists(golden_path):
        raise FileNotFoundError(f"CPU golden file not found: {golden_path}")

    output = read_matrix(output_path, dtype)
    golden = read_matrix(golden_path, dtype)

    expected_size = m * n
    if output.numel() != expected_size:
        raise ValueError(
            f"[Rank {rank_id}] npu output size {output.numel()} != expected size {expected_size}"
        )
    if golden.numel() != expected_size:
        raise ValueError(
            f"[Rank {rank_id}] cpu output size {golden.numel()} != expected size {expected_size}"
        )

    npu_output_tensor = output.reshape(m, n)
    golden_tensor = golden.reshape(m, n)

    numel = m * n
    if numel <= FULL_TENSOR_PRINT_MAX_ELEMENTS:
        print(f"\n[Rank {rank_id}] cpu golden:\n", golden_tensor)
        print(f"[Rank {rank_id}] npu output:\n", npu_output_tensor)
    else:
        _print_large_tensor_summary(golden_tensor, npu_output_tensor, m, n, rank_id)

    # 双千分之一：相对误差 > ERROR_TOL 的元素占比 ≤ FAIL_RATE_LIMIT 算过。
    # 注意 NaN/Inf：NPU 输出与 golden 的非有限值必须一致（同号 inf 或均为 nan），
    # 否则 NaN 参与运算会使 diff 为 NaN、diff > thr 恒为 False，导致全 NaN 输出被误判为 0% 失败。
    # 因此先构建“非有限值不一致”失败掩码，再叠加有限值的容差判断。
    g = golden_tensor.float()
    p = npu_output_tensor.float()
    both_finite = torch.isfinite(g) & torch.isfinite(p)
    same_nonfinite = (g == p) | (
        torch.isnan(g) & torch.isnan(p)
    )  # inf==inf / 双 nan 视为一致
    nonfinite_mismatch = (~both_finite) & (~same_nonfinite)

    abs_err = (p - g).abs()
    thr = ERROR_TOL + ERROR_TOL * g.abs()
    fail_mask = nonfinite_mismatch | (both_finite & (abs_err > thr))

    numel = g.numel()
    fail_cnt = fail_mask.sum().item()
    fail_rate = fail_cnt / numel if numel > 0 else 0.0
    print(
        f"[Rank {rank_id}] 双千分之一: fail_rate={fail_rate * 100:.5f}% "
        f"({fail_cnt}/{numel}, limit {FAIL_RATE_LIMIT * 100}%)"
    )
    mismatch_cnt = nonfinite_mismatch.sum().item()
    if mismatch_cnt > 0:
        print(
            f"[Rank {rank_id}] [WARN] {mismatch_cnt} 个元素 NaN/Inf 不一致"
            f"（含 NPU 输出为 NaN 而 golden 为有限值的情况）"
        )
    return fail_rate <= FAIL_RATE_LIMIT


def verify_result(m, n, rank_num, base_dir="./output", dtype="bf16"):
    all_pass = True
    results = []

    print(f"\n{'=' * 60}")
    print(f"Verifying outputs for {rank_num} ranks")
    print(f"Matrix shape: M={m}, N={n}, dtype={dtype}")
    print("All ranks should produce the same result after All2All")
    print(f"{'=' * 60}")

    for rank_id in range(rank_num):
        try:
            res = verify_single_rank(m, n, rank_id, base_dir, dtype)
            results.append(
                (rank_id, bool(res), None if res else "fail_rate exceeded limit")
            )
            if res:
                print(f"[PASS] Rank {rank_id}: NPU results match CPU golden")
            else:
                print(
                    f"[FAIL] Rank {rank_id}: 双千分之一 fail_rate exceeded "
                    f"(ERROR_TOL={ERROR_TOL}, limit={FAIL_RATE_LIMIT})"
                )
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
    parser = argparse.ArgumentParser(
        description="Verify NPU AllToAllMatmul output against CPU golden.",
    )
    parser.add_argument("m", type=int, help="matrix M dimension")
    parser.add_argument("n", type=int, help="matrix N dimension")
    parser.add_argument("rank_num", type=int, help="number of ranks")
    parser.add_argument(
        "base_dir",
        nargs="?",
        default="./output",
        help="base directory of output files (default: ./output)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        help="output dtype: bf16 (default) | fp16",
    )
    args = parser.parse_args()

    if args.dtype not in ("fp16", "bf16"):
        print(f"ERROR: --dtype must be 'fp16' or 'bf16', got {args.dtype}")
        sys.exit(1)

    try:
        all_pass = verify_result(
            args.m, args.n, args.rank_num, args.base_dir, args.dtype
        )
        if not all_pass:
            raise ValueError("[ERROR] Some NPU results differ from CPU.\n")
        print(f"\n[ALL PASS] All {args.rank_num} ranks passed verification!\n")

    except Exception as e:
        print(e)
        sys.exit(1)
