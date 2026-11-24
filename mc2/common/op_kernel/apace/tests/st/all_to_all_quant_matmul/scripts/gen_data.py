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

import math
import os
import sys

import numpy as np
import torch


# ---------------- 非量化（fp16/bf16 + bias）----------------


def _target_dtype(dtype):
    return torch.bfloat16 if dtype == "bf16" else torch.float16


def write_tensor(t, path):
    """按 2 字节原始位写入 fp16/bf16 张量（bf16 无 numpy 类型，经 uint16 视图写）。"""
    if t.dtype == torch.bfloat16:
        t.contiguous().view(torch.uint16).numpy().tofile(path)
    else:
        t.numpy().tofile(path)


def gen_nonquant_data(
    m,
    k,
    n,
    rank_num,
    dtype="fp16",
    lm=0,
    seed=678,
    a_range=(-5.0, 5.0),
    b_range=(-10.0, 10.0),
    bias_range=(-3.0, 3.0),
):
    M_total = m * rank_num
    K = k
    N = n
    Ka = K // rank_num
    target = _target_dtype(dtype)

    if K % rank_num != 0:
        raise ValueError(f"K={K} must be divisible by rank_num={rank_num}")

    np.random.seed(seed)

    a_local_f32 = [
        np.random.uniform(a_range[0], a_range[1], (M_total, Ka)).astype(np.float32)
        for _ in range(rank_num)
    ]
    b_ori_f32 = np.random.uniform(b_range[0], b_range[1], (N, K)).astype(np.float32)
    bias_f32 = np.random.uniform(bias_range[0], bias_range[1], (N,)).astype(np.float32)

    # 量化到目标 dtype（kernel 输入与 golden 必须用同一份量化值）
    a_local_list = [torch.from_numpy(x).to(target) for x in a_local_f32]
    b_ori = torch.from_numpy(b_ori_f32).to(target)  # (N, K)
    # bias 量化到目标 dtype：kernel 侧 bias 为量化值经 BiasTable 转 fp32，
    # golden 必须使用同一份量化值，不能用未量化的 fp32 bias
    bias = torch.from_numpy(bias_f32).to(target)  # (N,) 目标 dtype

    # all-to-all：dst 收集各 src 的 chunk
    #   a_after_alltoall[dst][src*m:(src+1)*m, :] = a_local[src][dst*m:(dst+1)*m, :]
    a_after_alltoall_list = []
    for dst_rank in range(rank_num):
        a_after = torch.zeros(M_total, Ka, dtype=torch.float32)
        for src_rank in range(rank_num):
            chunk = a_local_list[src_rank][dst_rank * m : (dst_rank + 1) * m, :].float()
            a_after[src_rank * m : (src_rank + 1) * m, :] = chunk
        a_after_alltoall_list.append(a_after)

    b_dequant = b_ori.float()  # (N, K)

    out_all_ranks = []
    for dst_rank in range(rank_num):
        a_after = a_after_alltoall_list[dst_rank]  # (M_total, Ka) fp32
        a_permuted = torch.zeros(m, K, dtype=torch.float32)
        for src_rank in range(rank_num):
            a_permuted[:, src_rank * Ka : (src_rank + 1) * Ka] = a_after[
                src_rank * m : (src_rank + 1) * m, :
            ]

        a_cpu = a_permuted  # (m, K)
        b_cpu = b_dequant  # (N, K)，逻辑 B = [K,N] 取转置
        out = torch.matmul(a_cpu, b_cpu.t())  # (m, N)
        out = out + bias.float()

        if lm == 2:
            # lm=2（GM累加）：golden 按 kernel 实际算法模拟，用于验证 kernel 机制正确性。
            # kernel：LOCAL 在 L0C fp32 算 self部分和+bias → 取整成输出 dtype 写 GM；
            # REMOTE 把其余 rank 部分和在 fp32 累加 → 原子加（取整）到 GM。
            # 故：self_q=round(self+bias); remote_q=round(Σ_remote); out=round(self_q+remote_q)。
            self_partial = torch.zeros(m, N, dtype=torch.float32)
            for src_rank in range(rank_num):
                if src_rank != dst_rank:
                    continue
                seg_a = a_after[src_rank * m : (src_rank + 1) * m, :]
                seg_b = b_dequant[:, src_rank * Ka : (src_rank + 1) * Ka]
                self_partial = self_partial + torch.matmul(seg_a, seg_b.t())

            remote_partial = torch.zeros(m, N, dtype=torch.float32)
            for src_rank in range(rank_num):
                if src_rank == dst_rank:
                    continue
                seg_a = a_after[src_rank * m : (src_rank + 1) * m, :]
                seg_b = b_dequant[:, src_rank * Ka : (src_rank + 1) * Ka]
                remote_partial = remote_partial + torch.matmul(seg_a, seg_b.t())

            self_q = (self_partial + bias.float()).to(target)
            remote_q = remote_partial.to(target)
            # 显式 .to(target)：保证输出为 fp16/bf16 写盘
            out = (self_q + remote_q).to(target)
        else:
            out = out.to(target)

        out_all_ranks.append(out)

    current_dir = os.getcwd()
    for rank_id in range(rank_num):
        input_dir = os.path.join(current_dir, "input", str(rank_id))
        output_dir = os.path.join(current_dir, "output", str(rank_id))
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        # a_local: (M_total, Ka) fp16/bf16，ND 行主序
        write_tensor(a_local_list[rank_id], os.path.join(input_dir, "input_a.bin"))
        # b_ori: (N, K)。kernel TransB=true 用 DN 列主序读 [K,N]，
        # host 写 [N,K] C-order 即等价于 [K,N] 列主序，故直接写 b_ori 不转置。
        write_tensor(b_ori, os.path.join(input_dir, "input_b.bin"))
        # bias: (N,) 目标 dtype（fp16/bf16 各 2 字节，与 kernel BiasType 一致）
        write_tensor(bias, os.path.join(input_dir, "input_bias.bin"))
        write_tensor(out_all_ranks[rank_id], os.path.join(output_dir, "cpu_output.bin"))


# ---------------- 量化（fp8 E4M3 + scale，输出 bf16）----------------


def gen_quant_data(m, k, n, rank_num):
    from en_dtypes import float8_e8m0
    from ml_dtypes import float8_e4m3fn

    M_total = m * rank_num
    K = k
    N = n
    Ka = K // rank_num

    if K % rank_num != 0:
        raise ValueError(f"K={K} must be divisible by rank_num={rank_num}")

    np.random.seed(42)

    b_ori = np.random.uniform(1, 8, (N, K)).astype(float8_e4m3fn)
    b_ori_transpose = np.swapaxes(b_ori, -1, -2)

    b_scale_ori = np.random.uniform(1, 8, size=(N, math.ceil(K / 64), 2)).astype(
        float8_e8m0
    )

    b_scale_reshape = b_scale_ori.reshape(N, -1)
    b_scale_broadcast = np.repeat(b_scale_reshape, 32, axis=-1)[..., :K]
    b_scale_broadcast_transpose = np.swapaxes(b_scale_broadcast, -1, -2)
    b_dequant = b_ori_transpose.astype(np.float32) * b_scale_broadcast_transpose.astype(
        np.float32
    )

    # [FIX] 2026-06-22: TransB=true 时 kernel 用 ScaleBDNLayoutPtn (column-major)
    # 访问 scaleB，host 必须写成 [N, K/32] C-order (= [K/32, N] 列主序) 才能被正确解读。
    b_scale_all = b_scale_ori.reshape(N, K // 32).flatten()

    a_local_list = []
    a_scale_list = []
    for rank_id in range(rank_num):
        a_local = np.random.uniform(1, 8, (M_total, Ka)).astype(float8_e4m3fn)
        a_scale = np.random.uniform(1, 8, size=(M_total, math.ceil(Ka / 64), 2)).astype(
            float8_e8m0
        )
        a_local_list.append(a_local)
        a_scale_list.append(a_scale)

    a_dequant_list = []
    for rank_id in range(rank_num):
        a_local = a_local_list[rank_id]
        a_scale = a_scale_list[rank_id]
        a_scale_reshape = a_scale.reshape(M_total, -1)
        a_scale_broadcast = np.repeat(a_scale_reshape, 32, axis=-1)[..., :Ka]
        a_dequant = a_local.astype(np.float32) * a_scale_broadcast.astype(np.float32)
        a_dequant_list.append(a_dequant)

    a_after_alltoall_list = []
    for dst_rank in range(rank_num):
        a_after_alltoall = np.zeros((M_total, Ka), dtype=np.float32)
        for src_rank in range(rank_num):
            chunk_row_start = dst_rank * m
            chunk_row_end = chunk_row_start + m
            chunk = a_dequant_list[src_rank][chunk_row_start:chunk_row_end, :]

            dst_row_start = src_rank * m
            dst_row_end = dst_row_start + m
            a_after_alltoall[dst_row_start:dst_row_end, :] = chunk
        a_after_alltoall_list.append(a_after_alltoall)

    out_all_ranks = []
    for dst_rank in range(rank_num):
        a_permuted = np.zeros((m, K), dtype=np.float32)
        for src_rank in range(rank_num):
            src_row_start = src_rank * m
            src_row_end = src_row_start + m
            src_chunk = a_after_alltoall_list[dst_rank][src_row_start:src_row_end, :]

            dst_col_start = src_rank * Ka
            dst_col_end = dst_col_start + Ka
            a_permuted[:, dst_col_start:dst_col_end] = src_chunk

        a_cpu = torch.from_numpy(a_permuted)
        b_cpu = torch.from_numpy(b_dequant)
        out = torch.matmul(a_cpu, b_cpu).to(torch.bfloat16)
        out_all_ranks.append(out)

    current_dir = os.getcwd()
    for rank_id in range(rank_num):
        input_dir = os.path.join(current_dir, "input", str(rank_id))
        output_dir = os.path.join(current_dir, "output", str(rank_id))
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        a_local_list[rank_id].view(np.uint8).tofile(
            os.path.join(input_dir, "input_a.bin")
        )
        b_ori.view(np.uint8).tofile(os.path.join(input_dir, "input_b.bin"))
        a_scale_list[rank_id].tofile(os.path.join(input_dir, "input_scaleA.bin"))
        b_scale_all.tofile(os.path.join(input_dir, "input_scaleB.bin"))
        out_all_ranks[rank_id].view(torch.uint16).numpy().tofile(
            os.path.join(output_dir, "cpu_output.bin")
        )


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python3 gen_data.py m k n rank_num [type] [dtype] [lm] [seed]")
        print("  m: matrix M dimension (per-rank output rows)")
        print("  k: matrix K dimension (total, must be divisible by rank_num)")
        print("  n: matrix N dimension")
        print("  rank_num: number of ranks")
        print("  type: optional, quant (default) | nonquant")
        print("  dtype: optional (nonquant), fp16 (default) | bf16")
        print(
            "  lm: optional (nonquant), localMatmul mode (0 default | 1 DEFERRED_SYNC | 2 GM累加)"
        )
        print("  seed: optional (nonquant), random seed (default 678)")
        sys.exit(1)

    m = int(sys.argv[1])
    k = int(sys.argv[2])
    n = int(sys.argv[3])
    rank_num = int(sys.argv[4])
    data_type = sys.argv[5] if len(sys.argv) >= 6 else "quant"
    if data_type not in ("quant", "nonquant"):
        print(f"ERROR: type must be 'quant' or 'nonquant', got {data_type}")
        sys.exit(1)

    if data_type == "quant":
        gen_quant_data(m, k, n, rank_num)
    else:
        dtype = sys.argv[6] if len(sys.argv) >= 7 else "fp16"
        if dtype not in ("fp16", "bf16"):
            print(f"ERROR: dtype must be 'fp16' or 'bf16', got {dtype}")
            sys.exit(1)
        lm = int(sys.argv[7]) if len(sys.argv) >= 8 else 0
        if lm not in (0, 1, 2):
            print(f"ERROR: lm must be 0/1/2, got {lm}")
            sys.exit(1)
        seed = int(sys.argv[8]) if len(sys.argv) >= 9 else 678
        gen_nonquant_data(m, k, n, rank_num, dtype, lm, seed)
