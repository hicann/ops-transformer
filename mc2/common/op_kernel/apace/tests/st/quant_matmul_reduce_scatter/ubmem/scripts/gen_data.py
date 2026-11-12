#!/usr/bin/python3
# coding=utf-8

# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""
Generate MX test data for QuantMatmulReduceScatter operator (dtype 泛化版).

Operator semantics: MX MatMul + AlltoAll + ReduceAdd.

dtype 泛化（对齐 experimental/mc2/quant_matmul_reduce_scatter）：
  - x1/x2: fp8_e4m3 | fp8_e5m2 | fp4_e2m1（fp8 可异构；fp4 仅 fp4+fp4 且仅 DN/ND）
  - y: bf16 | fp16
  - x2 layout: dn | nd | zn | nz（DN/ZN 转置 B [N,K]；ND/NZ 非转置 [K,N]）

Golden formula for rank j:
  C_total = sum_{i=0}^{R-1} (dequant(A_i) @ dequant(B_i))   # [M, N]
  y_j = C_total[j*M/R : (j+1)*M/R, :]                        # [M/R, N] (y dtype)

Usage:
  python3 gen_data.py m k n rank_num [x1_dtype x2_dtype y_dtype layout]
  # layout: dn|nd|zn|nz；x1_dtype/x2_dtype: e4m3|e5m2|fp4；y_dtype: bf16|fp16
"""

import math
import os
import sys

import numpy as np
import torch

FP4_MAGS = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
MXFP_DIVISOR = 64
MX_GROUP_SIZE = 32


def to_weight_nz_layout(b_k_n):
    """[*, *] uint8 → NZ blocked {ceil(col/32), ceil(row/16), 16, 32}."""
    rows, cols = b_k_n.shape
    rt = math.ceil(rows / 16)
    ct = math.ceil(cols / 32)
    padded = np.zeros((rt * 16, ct * 32), dtype=b_k_n.dtype)
    padded[:rows, :cols] = b_k_n
    blocked = padded.reshape(rt, 16, ct, 32)
    return np.ascontiguousarray(blocked.transpose(2, 0, 1, 3))


# ─────────────────────────────────────────────────────────────────────
#  dtype 量化 / 反量化
# ─────────────────────────────────────────────────────────────────────
def float8_e4m3fn_to_float(data_uint8):
    data = data_uint8.astype(np.uint8)
    sign = (data >> 7) & 1
    exp = (data >> 3) & 15
    mant = data & 7
    fp32 = np.zeros_like(data, dtype=np.float32)
    zero_mask = (exp == 0) & (mant == 0)
    fp32[zero_mask] = 0.0
    normal_mask = ~zero_mask
    fp32[normal_mask] = (
        (1.0 - 2.0 * sign[normal_mask])
        * np.power(2.0, exp[normal_mask] - 7.0)
        * (1.0 + mant[normal_mask] / 8.0)
    )
    return fp32


def float8_e5m2_to_float(data_uint8):
    data = data_uint8.astype(np.uint8)
    sign = (data >> 7) & 1
    exp = (data >> 2) & 31
    mant = data & 3
    fp32 = np.zeros_like(data, dtype=np.float32)
    zero_mask = (exp == 0) & (mant == 0)
    fp32[zero_mask] = 0.0
    normal_mask = (exp > 0) & (exp < 31)
    fp32[normal_mask] = (
        (1.0 - 2.0 * sign[normal_mask])
        * np.power(2.0, exp[normal_mask] - 15.0)
        * (1.0 + mant[normal_mask] / 4.0)
    )
    sub_mask = (exp == 0) & (mant != 0)
    fp32[sub_mask] = (
        (1.0 - 2.0 * sign[sub_mask]) * np.power(2.0, -14.0) * (mant[sub_mask] / 4.0)
    )
    return fp32


def float8_e8m0_to_float(scale_uint8):
    return np.power(2.0, scale_uint8.astype(np.float32) - 127.0)


def float_to_fp8_e4m3fn_vec(fp32_arr):
    fp32 = fp32_arr.astype(np.float32)
    sign = (fp32 < 0).astype(np.uint8)
    abs_val = np.abs(fp32)
    MAX_NORMAL = 1.875 * 128.0
    MIN_NORMAL = 2.0 ** (-6)
    log2_val = np.log2(np.clip(abs_val, 1e-30, None))
    e = np.floor(log2_val).astype(np.int32) + 7
    e = np.clip(e, 1, 14)
    normalized = abs_val / np.power(2.0, (e - 7).astype(np.float32))
    mant = np.clip(np.round((normalized - 1.0) * 8.0).astype(np.int32), 0, 7)
    overflow = abs_val >= MAX_NORMAL
    underflow = abs_val < MIN_NORMAL
    e[overflow] = 14
    mant[overflow] = 7
    e[underflow] = 0
    mant[underflow] = 0
    sign[underflow] = 0
    result = (sign << 7) | (e.astype(np.uint8) << 3) | mant.astype(np.uint8)
    result[fp32 == 0.0] = 0
    return result


def float_to_fp8_e5m2_vec(fp32_arr):
    fp32 = fp32_arr.astype(np.float32)
    sign = (fp32 < 0).astype(np.uint8)
    abs_val = np.abs(fp32)
    MAX_NORMAL = 1.75 * 2.0**15
    MIN_NORMAL = 2.0 ** (-14)
    log2_val = np.log2(np.clip(abs_val, 1e-30, None))
    e = np.floor(log2_val).astype(np.int32) + 15
    e = np.clip(e, 1, 30)
    normalized = abs_val / np.power(2.0, (e - 15).astype(np.float32))
    mant = np.clip(np.round((normalized - 1.0) * 4.0).astype(np.int32), 0, 3)
    overflow = abs_val >= MAX_NORMAL
    underflow = abs_val < MIN_NORMAL
    e[overflow] = 30
    mant[overflow] = 3
    e[underflow] = 0
    mant[underflow] = 0
    sign[underflow] = 0
    result = (sign << 7) | (e.astype(np.uint8) << 2) | mant.astype(np.uint8)
    result[fp32 == 0.0] = 0
    return result


def float_to_fp4_e2m1_vec(fp32_arr):
    fp32 = fp32_arr.astype(np.float32)
    sign = (fp32 < 0).astype(np.uint8)
    abs_val = np.abs(fp32)
    flat = abs_val.reshape(-1)
    codes = np.empty(flat.shape, dtype=np.uint8)
    for s in range(0, flat.size, 1 << 20):
        blk = flat[s : s + (1 << 20)]
        codes[s : s + (1 << 20)] = np.argmin(np.abs(blk[:, None] - FP4_MAGS), axis=-1)
    return (sign << 3) | codes.reshape(abs_val.shape)


def fp4_e2m1_to_float(codes):
    return (1.0 - 2.0 * ((codes >> 3) & 1)) * FP4_MAGS[(codes & 7).astype(np.int64)]


def pack_fp4x2(codes):
    return ((codes[..., 0::2] & 0xF) | ((codes[..., 1::2] & 0xF) << 4)).astype(np.uint8)


def unpack_fp4x2(packed):
    lo = (packed & 0xF).astype(np.uint8)
    hi = ((packed >> 4) & 0xF).astype(np.uint8)
    out = np.empty(packed.shape[:-1] + (packed.shape[-1] * 2,), dtype=np.uint8)
    out[..., 0::2] = lo
    out[..., 1::2] = hi
    return out


def quantize(data_fp32, dtype):
    """FP32 → 存储字节（fp8 1B/元素；fp4 打包 2 元素/字节）。"""
    if dtype == "fp8_e4m3":
        return float_to_fp8_e4m3fn_vec(data_fp32)
    if dtype == "fp8_e5m2":
        return float_to_fp8_e5m2_vec(data_fp32)
    if dtype == "fp4_e2m1":
        return pack_fp4x2(float_to_fp4_e2m1_vec(data_fp32))
    raise ValueError(f"unknown dtype: {dtype}")


def dequantize(data_bytes, dtype):
    """存储字节 → FP32（fp8 直接反量化；fp4 解包后反量化）。"""
    if dtype == "fp8_e4m3":
        return float8_e4m3fn_to_float(data_bytes.astype(np.uint8))
    if dtype == "fp8_e5m2":
        return float8_e5m2_to_float(data_bytes.astype(np.uint8))
    if dtype == "fp4_e2m1":
        return fp4_e2m1_to_float(unpack_fp4x2(data_bytes.astype(np.uint8)))
    raise ValueError(f"unknown dtype: {dtype}")


def dequant_with_scale(data_bytes, dtype, scale_u8, is_trans_b):
    """按 (dtype, transB) 反量化 A 或 B 并乘 e8m0 scale。

    scale_u8 语义（MX 每 64 元素 2 个 scale，每 32 元素一组，沿 K 维）：
      A:  [M, kd, 2]（dq [M,K]）
      B 转置(DN/ZN): [n, kd, 2]（dq [N,K]）
      B 非转置(ND/NZ): [kd, n, 2]（dq [K,N]，需沿 K 维展开）
    """
    dq = dequantize(data_bytes, dtype)
    sf = float8_e8m0_to_float(scale_u8.astype(np.uint8))
    k = dq.shape[1]
    if is_trans_b:
        # B 转置：[n, kd, 2] → 沿 K（dim1）每 32 元素一组重复 → [n, K]
        sf_full = np.repeat(sf, MX_GROUP_SIZE, axis=-1).reshape(sf.shape[0], -1)[:, :k]
        return dq * sf_full
    if scale_u8.ndim == 3 and scale_u8.shape[0] == dq.shape[0]:
        # A：[M, kd, 2] → [M, K]（沿 K 展开）
        sf_full = np.repeat(sf, MX_GROUP_SIZE, axis=-1).reshape(sf.shape[0], -1)[:, :k]
        return dq * sf_full
    # B 非转置：[kd, n, 2] → 沿 K（dim0）展开 → [K, N]
    sf_full = (
        np.repeat(sf, MX_GROUP_SIZE, axis=-1)
        .transpose(0, 2, 1)
        .reshape(-1, dq.shape[1])[:k, :]
    )
    return dq * sf_full


def write_artifacts(base_dir, rank_id, a_bytes, b_bytes, a_scale, b_scale, out):
    input_dir = os.path.join(base_dir, "input", str(rank_id))
    output_dir = os.path.join(base_dir, "output", str(rank_id))
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    a_bytes.tofile(os.path.join(input_dir, "input_a.bin"))
    b_bytes.tofile(os.path.join(input_dir, "input_b.bin"))
    a_scale.tofile(os.path.join(input_dir, "input_scaleA.bin"))
    b_scale.tofile(os.path.join(input_dir, "input_scaleB.bin"))
    out.view(torch.uint16).numpy().tofile(os.path.join(output_dir, "cpu_output.bin"))


def gen_golden_data_quant_matmul_reduce_scatter(
    m, k, n, rank_num, x1_dtype, x2_dtype, y_dtype, layout
):
    M, K, N = m, k, n
    tp_size_m = M // rank_num
    is_trans_b = layout in ("dn", "zn")
    kd = math.ceil(K / MXFP_DIVISOR)
    is_fp4 = x1_dtype == "fp4_e2m1"
    # fp16 输出时缩小数据/scale 范围避免溢出
    small = y_dtype == "fp16"
    data_lo, data_hi = (1.0, 2.0) if small else (1.0, 8.0)
    scale_lo, scale_hi = (127, 128) if small else (127, 129)

    print(
        f"  M={M}, K={K}, N={N}, rankNum={rank_num}, x1={x1_dtype}, x2={x2_dtype}, y={y_dtype}, layout={layout}"
    )

    a_list, a_scale_list, b_list, b_scale_list = [], [], [], []
    for rank_id in range(rank_num):
        np.random.seed(42 + rank_id)
        a_f32 = np.random.uniform(data_lo, data_hi, (M, K)).astype(np.float32)
        a_bytes = quantize(a_f32, x1_dtype)
        a_scale = np.random.randint(scale_lo, scale_hi, (M, kd, 2)).astype(np.uint8)
        if is_trans_b:
            # B 逻辑 [N, K]，scale [n, kd, 2]
            b_f32 = np.random.uniform(data_lo, data_hi, (N, K)).astype(np.float32)
            b_scale = np.random.randint(scale_lo, scale_hi, (N, kd, 2)).astype(np.uint8)
        else:
            b_f32 = np.random.uniform(data_lo, data_hi, (K, N)).astype(np.float32)
            b_scale = np.random.randint(scale_lo, scale_hi, (kd, N, 2)).astype(np.uint8)
        b_bytes = quantize(b_f32, x2_dtype)
        a_list.append(a_bytes)
        a_scale_list.append(a_scale)
        b_list.append(b_bytes)
        b_scale_list.append(b_scale)

    # B 存储字节按 layout 重排
    b_storage = []
    for rank_id in range(rank_num):
        b_bytes = b_list[rank_id]  # 逻辑字节：[N,K/2]（DN/fp4）或 [K,N/2]（ND/fp4）
        if layout == "dn":
            b_storage.append(np.ascontiguousarray(b_bytes))  # [N,K] 连续
        elif layout == "nd":
            b_storage.append(np.ascontiguousarray(b_bytes))  # [K,N] 连续
        elif layout == "zn":
            b_storage.append(to_weight_nz_layout(b_bytes))  # [N,K] → {Kt,Nt,16,32}
        else:  # nz
            b_storage.append(to_weight_nz_layout(b_bytes))  # [K,N] → {Nt,Kt,16,32}

    # golden: C_total = sum_i(dequant(A_i) @ dequant(B_i))，FP32
    c_total = np.zeros((M, N), dtype=np.float32)
    for rank_id in range(rank_num):
        a_deq = dequant_with_scale(
            a_list[rank_id], x1_dtype, a_scale_list[rank_id], False
        )
        b_deq = dequant_with_scale(
            b_list[rank_id], x2_dtype, b_scale_list[rank_id], is_trans_b
        )
        a_cpu = torch.from_numpy(a_deq)
        b_cpu = torch.from_numpy(b_deq)
        if is_trans_b:
            # B 逻辑 [N, K] → matmul(A, B^T)
            c_total += torch.matmul(a_cpu, b_cpu.t()).numpy()
        else:
            c_total += torch.matmul(a_cpu, b_cpu).numpy()

    c_total_tensor = torch.from_numpy(c_total)
    y_torch = {"bf16": torch.bfloat16, "fp16": torch.float16}[y_dtype]

    base_dir = os.getcwd()
    for rank_id in range(rank_num):
        y_j = c_total_tensor[rank_id * tp_size_m : (rank_id + 1) * tp_size_m, :]
        y_out = y_j.to(y_torch)
        write_artifacts(
            base_dir,
            rank_id,
            a_list[rank_id],
            b_storage[rank_id],
            a_scale_list[rank_id],
            b_scale_list[rank_id],
            y_out,
        )


def main():
    usage = (
        "Usage: python3 gen_data.py m k n rank_num [x1_dtype x2_dtype y_dtype layout]"
    )
    if len(sys.argv) < 5:
        print(usage)
        print("  x1_dtype/x2_dtype: e4m3|e5m2|fp4 (default e4m3)")
        print("  y_dtype: bf16|fp16 (default bf16)")
        print("  layout: dn|nd|zn|nz (default nz)")
        sys.exit(1)

    m = int(sys.argv[1])
    k = int(sys.argv[2])
    n = int(sys.argv[3])
    rank_num = int(sys.argv[4])
    x1_dtype = sys.argv[5] if len(sys.argv) >= 6 else "fp8_e4m3"
    x2_dtype = sys.argv[6] if len(sys.argv) >= 7 else "fp8_e4m3"
    y_dtype = sys.argv[7] if len(sys.argv) >= 8 else "bf16"
    layout = sys.argv[8] if len(sys.argv) >= 9 else "nz"

    if x1_dtype not in ("fp8_e4m3", "fp8_e5m2", "fp4_e2m1") or x2_dtype not in (
        "fp8_e4m3",
        "fp8_e5m2",
        "fp4_e2m1",
    ):
        print(f"Error: invalid dtype {x1_dtype}/{x2_dtype}")
        sys.exit(1)
    if y_dtype not in ("bf16", "fp16"):
        print(f"Error: invalid y_dtype {y_dtype}")
        sys.exit(1)
    if layout not in ("dn", "nd", "zn", "nz"):
        print(f"Error: invalid layout {layout}")
        sys.exit(1)
    if (x1_dtype == "fp4_e2m1") != (x2_dtype == "fp4_e2m1"):
        print("Error: FP4 requires both x1 and x2 to be fp4")
        sys.exit(1)
    if x1_dtype == "fp4_e2m1" and layout in ("zn", "nz"):
        print("Error: FP4 does not support ZN/NZ layout")
        sys.exit(1)

    if k % 32 != 0:
        print(f"Error: K={k} not divisible by 32")
        sys.exit(1)
    if math.ceil(k / 64) % 2 != 0:
        print(f"Error: ceil(K/64)={math.ceil(k / 64)} not even")
        sys.exit(1)
    if m % rank_num != 0:
        print(f"Error: M={m} not divisible by rank_num={rank_num}")
        sys.exit(1)

    gen_golden_data_quant_matmul_reduce_scatter(
        m, k, n, rank_num, x1_dtype, x2_dtype, y_dtype, layout
    )
    print("Test data generation completed!")


if __name__ == "__main__":
    main()
