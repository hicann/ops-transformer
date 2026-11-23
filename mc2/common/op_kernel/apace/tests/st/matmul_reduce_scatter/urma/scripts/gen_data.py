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
# quantMatmulReduceScatter golden 数据生成
#
# 每 rank 输入 A 为 ND [m,k], B 为 DN [n,k] 且所有 rank 共享.
# ScaleA/ScaleB 的形状分别为 [m,ceil(k/64),2] 和 [n,ceil(k/64),2], 类型为 E8M0,
# 每个 scale 覆盖 32 个 K 元素; MXFP4 沿 K 按 2 元素/字节打包, 偶数元素位于低半字节.
# 计算流程:
#   1. 每卡独立计算 C_i = bf16(dequant(A_i) @ dequant(B)) -> [m,n]
#   2. AllToAll 按 M 轴切分 C_i, chunk j 分发到 rank j
#   3. reduceSum: output_r = Σ_{j=0}^{R-1} C_j[chunk r] -> [m/rankSize,n]
# K 不切分 (每卡完整 K); 每卡 A 不同, B 相同
# ----------------------------------------------------------------------------------------------------------

import math
import os
import sys

import numpy as np
import torch
from en_dtypes import float8_e8m0
from ml_dtypes import float8_e4m3fn, float8_e5m2

# dtype 编号: 0=E2M1E2M1 (MXFP4, 2 元素/字节打包) 1=E4M3E4M3(默认) 2=E5M2E5M2 3=E4M3E5M2 4=E5M2E4M3
QUANT_TYPES = {
    0: ("e2m1", "e2m1"),
    1: ("e4m3", "e4m3"),
    2: ("e5m2", "e5m2"),
    3: ("e4m3", "e5m2"),
    4: ("e5m2", "e4m3"),
}
DTYPE_MAP = {"e4m3": float8_e4m3fn, "e5m2": float8_e5m2}

# ---------------- FP4 E2M1 (packed fp4x2, 与硬件同字节序) ----------------

FP4_MAGS = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)


def float_to_fp4_e2m1_vec(fp32_arr):
    """FP32 -> FP4 E2M1 element codes (uint8 nibble, bit3=sign, bits2-0=magnitude code)。"""
    fp32 = fp32_arr.astype(np.float32)
    sign = (fp32 < 0).astype(np.uint8)
    abs_val = np.abs(fp32)

    flat = abs_val.reshape(-1)
    codes = np.empty(flat.shape, dtype=np.uint8)
    chunk = 1 << 20
    for s in range(0, flat.size, chunk):
        blk = flat[s : s + chunk]
        idx = np.argmin(np.abs(blk[:, None] - FP4_MAGS), axis=-1)
        codes[s : s + chunk] = idx
    codes = codes.reshape(abs_val.shape)
    return (sign << 3) | codes


def fp4_e2m1_to_float(codes):
    """FP4 E2M1 element codes (nibble) -> FP32。"""
    sign = (codes >> 3) & 1
    mag = FP4_MAGS[(codes & 7).astype(np.int64)]
    return (1.0 - 2.0 * sign) * mag


def pack_fp4x2(codes):
    """沿最后一轴打包, 2 元素/字节 (低 nibble 在前, 对齐 fp4x2_e2m1_t 硬件序)。"""
    even = codes[..., 0::2] & 0xF
    odd = codes[..., 1::2] & 0xF
    return (even | (odd << 4)).astype(np.uint8)


def quantize_data(kind, shape):
    """生成量化数据: fp8 经 ml_dtypes cast; fp4 用最近邻 E2M1 编码 (返回 nibble codes)。"""
    if kind == "e2m1":
        return float_to_fp4_e2m1_vec(np.random.uniform(1.0, 6.0, shape))
    return np.random.uniform(1, 8, shape).astype(DTYPE_MAP[kind])


def quantized_to_file_bytes(kind, arr):
    """量化数据 -> 写文件字节: fp4 打包 2 元素/字节; fp8 原样。"""
    if kind == "e2m1":
        return pack_fp4x2(arr)
    return arr.view(np.uint8)


def dequantize_data(kind, arr):
    """量化数据 -> FP32 (乘 scale 前)。"""
    if kind == "e2m1":
        return fp4_e2m1_to_float(arr)
    return arr.astype(np.float32)


def write_artifacts(
    base_dir, rank_id, a_quant, b_quant, a_scale, b_scale, out, quant_type
):
    input_dir = os.path.join(base_dir, "input", str(rank_id))
    output_dir = os.path.join(base_dir, "output", str(rank_id))
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    kind_a, kind_b = QUANT_TYPES[quant_type]
    quantized_to_file_bytes(kind_a, a_quant).tofile(
        os.path.join(input_dir, "input_a.bin")
    )
    quantized_to_file_bytes(kind_b, b_quant).tofile(
        os.path.join(input_dir, "input_b.bin")
    )
    a_scale.tofile(os.path.join(input_dir, "input_scaleA.bin"))
    b_scale.tofile(os.path.join(input_dir, "input_scaleB.bin"))
    # dtype 元数据: main.cpp 读入与 argv 交叉校验
    with open(os.path.join(input_dir, "quant_type.txt"), "w") as f:
        f.write(f"{quant_type}\n")
    out.view(torch.uint16).numpy().tofile(os.path.join(output_dir, "cpu_output.bin"))


def gen_golden_data_quant_matmul_reduce_scatter(m, k, n, rank_num, quant_type=1):
    """
    生成 quantMatmulReduceScatter 的 golden 数据.

    参数:
      m: 矩阵 A 的行数, 必须为正数且能被 rank_num 整除
      k: 矩阵 A 的列数, 必须为完整 K 且 ceil(k/64) 为偶数; MXFP4 时 k 还必须为偶数
      n: 矩阵 B 的列数, 必须为正数
      rank_num: rank 数量, 范围为 [2,16]
      quant_type: A/B 量化类型组合编号 (QUANT_TYPES, 默认 1=E4M3E4M3)
    """
    if m <= 0 or k <= 0 or n <= 0:
        raise ValueError(f"m, k and n must be positive, got m={m}, k={k}, n={n}")
    if rank_num < 2 or rank_num > 16:
        raise ValueError(f"rank_num must be in [2, 16], got {rank_num}")
    if math.ceil(k / 64) % 2 != 0:
        raise ValueError(f"ceil(k/64) must be even, got k={k}")
    if m % rank_num != 0:
        raise ValueError(f"m={m} must be divisible by rank_num={rank_num}")
    if quant_type not in QUANT_TYPES:
        raise ValueError(
            f"quant_type must be one of {sorted(QUANT_TYPES.keys())}, got {quant_type}"
        )

    kind_a, kind_b = QUANT_TYPES[quant_type]
    if "e2m1" in (kind_a, kind_b) and k % 2 != 0:
        raise ValueError(
            f"MXFP4 requires even k for 2-elements-per-byte packing, got k={k}"
        )

    chunk_m = m // rank_num  # 每个 rank 的 M 分片大小

    np.random.seed(42)

    # B 的逻辑形状为 [k,n], 以 DN [n,k] 行主序存储, 所有 rank 共享
    b_ori = quantize_data(kind_b, (n, k))

    # B Scale: [n, ceil(k/64), 2] → [n, 2*ceil(k/64)] 展平
    b_scale_ori = np.random.uniform(1, 8, size=(n, math.ceil(k / 64), 2)).astype(
        float8_e8m0
    )
    b_scale_all = b_scale_ori.reshape(n, math.ceil(k / 64) * 2).flatten()

    # scale 每 32 个 K 元素广播一次, 尾部截断到 k; B 反量化后转置为 [k,n]
    k_padded = math.ceil(k / 64) * 64
    b_scale_broadcast = b_scale_ori.reshape(n, math.ceil(k / 64), 2, 1)
    b_scale_broadcast = np.repeat(
        b_scale_broadcast, 32, axis=-1
    )  # [n, ceil(K/64), 2, 32]
    b_scale_broadcast = b_scale_broadcast.reshape(n, k_padded)[..., :k]  # [n, k]
    b_scale_broadcast_transpose = np.swapaxes(b_scale_broadcast, -1, -2)  # [k, n]
    b_dequant = dequantize_data(kind_b, b_ori).T * b_scale_broadcast_transpose  # [k, n]

    # 每卡独立的 A 矩阵
    a_list = []
    a_scale_list = []
    c_list = []

    for rank_id in range(rank_num):
        # A [m, k] 量化 (dtype 随 quant_type 的 A 侧)
        a_local = quantize_data(kind_a, (m, k))
        # ScaleA [m, ceil(k/64), 2] fp8_e8m0
        a_scale = np.random.uniform(1, 8, size=(m, math.ceil(k / 64), 2)).astype(
            float8_e8m0
        )

        a_list.append(a_local)
        a_scale_list.append(a_scale)

        # 反量化 A (scale 展开同 B 侧)
        a_scale_broadcast = a_scale.reshape(m, math.ceil(k / 64), 2, 1)
        a_scale_broadcast = np.repeat(
            a_scale_broadcast, 32, axis=-1
        )  # [m, ceil(K/64), 2, 32]
        a_scale_broadcast = a_scale_broadcast.reshape(m, k_padded)[..., :k]  # [m, k]
        a_dequant = dequantize_data(kind_a, a_local) * a_scale_broadcast  # [m, k]

        # C_i = bf16(dequant(A_i) @ dequant(B)) -> [m,n]
        a_cpu = torch.from_numpy(a_dequant)
        b_cpu = torch.from_numpy(b_dequant)
        c_i = torch.matmul(a_cpu, b_cpu).to(torch.bfloat16)  # [m, n]
        c_list.append(c_i)

    # output_r = Σ_{j=0}^{R-1} C_j[rows r*chunk_m : (r+1)*chunk_m, : ]
    # 各 C_i 已先转为 BF16; 跨 rank 求和时转回 float32 累加, 最后再转 BF16
    for dst_rank in range(rank_num):
        output = torch.zeros((chunk_m, n), dtype=torch.float32)
        for src_rank in range(rank_num):
            row_start = dst_rank * chunk_m
            row_end = row_start + chunk_m
            chunk = c_list[src_rank][row_start:row_end, :].float()
            output += chunk

        output = output.to(torch.bfloat16)

        current_dir = os.getcwd()
        write_artifacts(
            current_dir,
            dst_rank,
            a_list[dst_rank],
            b_ori,
            a_scale_list[dst_rank],
            b_scale_all,
            output,
            quant_type,
        )

    print(
        f"Generated golden data for m={m}, k={k}, n={n}, rank_num={rank_num}, quant_type={quant_type} "
        f"({kind_a}x{kind_b})"
    )
    print(f"  chunk_m={chunk_m}, output per rank: [{chunk_m}, {n}]")


if __name__ == "__main__":
    if len(sys.argv) < 5 or len(sys.argv) > 6:
        print("Usage: python3 gen_data.py m k n rank_num [quant_type]")
        print("  m: matrix M dimension (must be divisible by rank_num)")
        print("  k: matrix K dimension (full K, NOT split across ranks)")
        print("  n: matrix N dimension")
        print("  rank_num: number of ranks")
        print(
            "  quant_type: 0=E2M1E2M1 (MXFP4) | 1=E4M3E4M3 (default) | 2=E5M2E5M2 | 3=E4M3E5M2 | 4=E5M2E4M3"
        )
        sys.exit(1)

    m = int(sys.argv[1])
    k = int(sys.argv[2])
    n = int(sys.argv[3])
    rank_num = int(sys.argv[4])
    quant_type = int(sys.argv[5]) if len(sys.argv) == 6 else 1

    gen_golden_data_quant_matmul_reduce_scatter(m, k, n, rank_num, quant_type)
