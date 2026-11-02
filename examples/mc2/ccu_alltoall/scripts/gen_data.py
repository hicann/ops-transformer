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

"""CCU AllToAll + MXFP8 MatMul 示例数据生成器（纯 stdlib）。

位型闭式与原 apace ST C++ 生成器（test_data_utils.h）逐位一致：
  FP8_E4M3FN_POW2 = [0x38, 0x40, 0x48, 0x50]（2^0~2^3 的 E4M3FN 编码）
  enc4(v) = FP8_E4M3FN_POW2[v % 4]；enc2(v) = 偶 -> 0x38 / 奇 -> 0x40

数据语义（K-并行 GEMM）：x1=[M,K] FP8 E4M3FN（token 轴均分 R 块做 A2A），x2=[R*K,N]
完整权重不通信，Y=[M/R,N] BF16 = sum_r recvData[r块,:] @ x2[r*K段,:]。

产物（--out 目录，默认 data；scaleKLen = K//64*2，blockRows = M//R）：
  meta.txt              一行 "M K N R"
  x1_rank{s}.bin        [M,K]：行 (j*blockRows+m) 整行常量 enc4(2*j+s+m)
  x1_scale_rank{s}.bin  M*scaleKLen 字节 0x7F（E8M0 1.0）
  x2.bin                [R*K,N] DN 列主序：字节 (k + n*R*K) = enc2(k//K)，文件 = 列 * N
  x2_scale.bin          R*scaleKLen*N 字节 0x7F
  y_expect_rank{s}.bin  [blockRows,N] BF16：行 m 整行 = expect_y_bf16_bits(s, m, R, K) 位型

用法：python3 gen_data.py --M 512 --K 256 --N 256 --ranks 4 --out data
"""

import argparse
import os
import struct

# FP8 E4M3FN / E8M0 常量编码（2 的幂，精确可表示；E8M0 127 = 1.0，dequant 恒等）
FP8_E4M3FN_POW2 = [0x38, 0x40, 0x48, 0x50]
FP8_E8M0_SCALE_1_0 = 127


def enc4(v):
    return FP8_E4M3FN_POW2[v % 4]


def enc2(v):
    return FP8_E4M3FN_POW2[0] if v % 2 == 0 else FP8_E4M3FN_POW2[1]


def expect_y_bf16_bits(my_rank, m, R, K):
    """C++ ExpectYBf16Bits 等价：Y[m] = K * sum_r 2^((2i+r+m)%4 + r%2)。

    各加数均为 2 的幂，和 <= 256（R<=16），乘 K 后为 <= 8 位有效数字的整数，
    float32 与 float64 均精确表示；取 float32 位型高 16 位（截断，与 C++ bits>>16 一致）。
    """
    total = 0.0
    for r in range(R):
        e = (2 * my_rank + r + m) % 4 + (r % 2)
        total += float(1 << e)
    return struct.unpack("<I", struct.pack("<f", float(K) * total))[0] >> 16


def main():
    parser = argparse.ArgumentParser(
        description="CCU AllToAll+MXFP8 MatMul 示例数据生成器（位型闭式，与原 C++ 生成器逐位一致）"
    )
    parser.add_argument(
        "--M", type=int, required=True, help="token 维总行数（须被 ranks 整除）"
    )
    parser.add_argument(
        "--K", type=int, required=True, help="reduce 维（须为 64 的倍数，MXFP8）"
    )
    parser.add_argument("--N", type=int, required=True, help="输出列数")
    parser.add_argument(
        "--ranks", type=int, required=True, help="rank 数（须与运行时 NPU 数一致）"
    )
    parser.add_argument("--out", default="data", help="输出目录（默认 data）")
    args = parser.parse_args()

    M, K, N, R = args.M, args.K, args.N, args.ranks
    if M < 1 or N < 1 or R < 1:
        parser.error("M / N / ranks must be positive")
    if M % R != 0:
        parser.error("M (%d) must be divisible by ranks (%d)" % (M, R))
    if K <= 0 or K % 64 != 0:
        parser.error("K (%d) must be positive and divisible by 64 (MXFP8)" % K)

    scale_k_len = K // 64 * 2
    block_rows = M // R

    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "meta.txt"), "w") as f:
        f.write("%d %d %d %d\n" % (M, K, N, R))

    # x1_rank{s}.bin [M,K]：行 (j*blockRows+m) 整行常量 enc4(2*j+s+m)
    for s in range(R):
        with open(os.path.join(out_dir, "x1_rank%d.bin" % s), "wb") as f:
            for j in range(R):
                for m in range(block_rows):
                    f.write(bytes([enc4(2 * j + s + m)]) * K)

    # x1_scale_rank{s}.bin：M*scaleKLen 字节 0x7F（E8M0 1.0）
    for s in range(R):
        with open(os.path.join(out_dir, "x1_scale_rank%d.bin" % s), "wb") as f:
            f.write(bytes([FP8_E8M0_SCALE_1_0]) * (M * scale_k_len))

    # x2.bin [R*K,N] DN 列主序：列 n = join(enc2(r)*K for r in R)，文件 = 列 * N
    with open(os.path.join(out_dir, "x2.bin"), "wb") as f:
        col = b"".join(bytes([enc2(r)]) * K for r in range(R))
        for _ in range(N):
            f.write(col)

    # x2_scale.bin：R*scaleKLen*N 字节 0x7F
    with open(os.path.join(out_dir, "x2_scale.bin"), "wb") as f:
        f.write(bytes([FP8_E8M0_SCALE_1_0]) * (R * scale_k_len * N))

    # y_expect_rank{s}.bin [blockRows,N] BF16：行 m 整行 = expect 位型
    for s in range(R):
        with open(os.path.join(out_dir, "y_expect_rank%d.bin" % s), "wb") as f:
            for m in range(block_rows):
                f.write(struct.pack("<H", expect_y_bf16_bits(s, m, R, K)) * N)

    print(
        "data generated in %s: M=%d K=%d N=%d ranks=%d (blockRows=%d, scaleKLen=%d)"
        % (out_dir, M, K, N, R, block_rows, scale_k_len)
    )


if __name__ == "__main__":
    main()
