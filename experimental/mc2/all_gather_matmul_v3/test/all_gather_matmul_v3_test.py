#!/usr/bin/env python3
# coding=utf-8
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, and you can redistribute it and/or modify it under the terms of
# CANN Open Software License Agreement Version 2.0 (the License).
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""AllGatherMatmulV3 (ubmem) 单文件自包含测试脚本：精度模式 + 性能模式。

语义（per rank i）:
  1. AllGather: A_all = concat([A_0, ..., A_{R-1}])  -> [R*M, K]
  2. Matmul:    Y_i = dequant(A_all) @ dequant(B_i)^T -> [R*M, N] (bf16/fp16)

dtype 组合通过 --x1-dtype/--x2-dtype(e4m3/e5m2/fp4) 与 --y-dtype(bf16/fp16) 指定,
fp8 任意混合(8 种)、fp4 x fp4(2 种) 共 10 种组合。

用法:
  # 精度模式（CPU golden 反量化 matmul 做参考，allclose 比对）
  python3 all_gather_matmul_v3_test.py --mode precision --cases all --world-size 4
  python3 all_gather_matmul_v3_test.py --mode precision --cases m4,m96 --world-size 4
  python3 all_gather_matmul_v3_test.py --mode precision --m 48 --k 3072 --n 2560   # 自定义形状

  # 性能模式（golden 小算子级联 vs mc2 融合算子，profiler 采集 + 通算掩盖率）
  python3 all_gather_matmul_v3_test.py --mode perf --cases all --world-size 4

  # 列出内置用例
  python3 all_gather_matmul_v3_test.py --list-cases

环境准备:
  source <CANN_HOME>/set_env.sh          # 安装了 custom_transformer 算子包的 CANN
  #   export HCCL_OP_EXPANSION_MODE=CCU_SCHED   # golden 小算子走 CCU 编排
  #   unset  HCCL_BUFFSIZE                      # ubmem 通路不要设置该变量

  多机/多卡通过 ASCEND_RT_VISIBLE_DEVICES 控制可见设备数（world-size 与之匹配）。

输出:
  精度: 逐轮打印误差统计与 PASS/FAIL, 追加写入 all_gather_matmul_v3_precision_result.csv
  性能: 打印 mc2/golden 耗时、掩盖率与末轮精度, 追加写入 all_gather_matmul_v3_performance_result.csv
        (列: case_name, world_size, mc2_prof(us), golden_prof(us), avg_hccl, avg_matmul, avg_aicpu,
         avg_trans, mask_rate, result, precision, timestamp)

测量:
  精度: golden 小算子级联预跑 --golden-iters 轮(默认 10, 先把内置 HCCL buffer 用脏) ->
        融合算子连续 --iters 轮(默认 10), 每轮都与 CPU golden(fp32 反量化 matmul)比对;
  性能: profiler 10 迭代(i==1 时插入重型算子防 host bound), op_summary 跳过 Exp 行,
        取后 5 轮平均、跨 rank 取 max; mask = (golden - mc2) / min(avg_hccl, avg_matmul + avg_trans),
        mask > 0.7 记 PASS; 性能跑完后附带末轮精度校验。
"""

import argparse
import csv
import math
import os
import re
import shutil
import sys
import time

import numpy as np
import torch
import torch_npu
import torch.distributed as dist
from ml_dtypes import float4_e2m1fn as np_fp4_e2m1
from ml_dtypes import float8_e4m3fn as np_fp8_e4m3fn
from ml_dtypes import float8_e5m2 as np_fp8_e5m2
from ml_dtypes import float8_e8m0fnu as np_fp8_e8m0fnu
from torch.multiprocessing import Process, Manager

import cann_ops_transformer

# ----------------------------------------------------------------------------- 常量
MASTER_IP = "127.0.0.1"
MASTER_PORT = 50220

ACL_BF16 = 15  # y_dtype 枚举
ACL_FP16 = 5  # y_dtype 枚举
ACL_FP8_E4M3FN = 24  # x1/x2 枚举
ACL_FP8_E5M2 = 23  # x1/x2 枚举
ACL_FP4_E2M1 = 296  # x1/x2 枚举(uint8 打包 fp4)
ACL_FP8_E8M0 = 293  # scale 枚举
MX_GROUP_SIZES = [1, 1, 32]
MX_GROUP_SIZE = 32

# 量化类型 -> (ml_dtypes 逻辑 dtype, torch 存储 dtype, ACL 枚举)
QUANT_KINDS = {
    "e4m3": (np_fp8_e4m3fn, torch.float8_e4m3fn, ACL_FP8_E4M3FN),
    "e5m2": (np_fp8_e5m2, torch.float8_e5m2, ACL_FP8_E5M2),
    "fp4": (np_fp4_e2m1, torch.uint8, ACL_FP4_E2M1),
}
OUT_KINDS = {"bf16": (torch.bfloat16, ACL_BF16), "fp16": (torch.float16, ACL_FP16)}

FIXED_SEED = 42  # 每 rank 种子 = FIXED_SEED + rank, 所有 rank 生成完全一致的输入
ERROR_TOL = 1e-2  # 精度比对 rtol=atol
PERF_ITERS = 10  # profiler 内迭代次数, 统计取后 5 次

# 18 个默认形状（TP4, W8A8 MX 量化, per-rank m=4~96, n=2560, k=3072）
DEFAULT_CASES = [
    # (case_name, m_per_rank, k, n)
    ("m4", 4, 3072, 2560),
    ("m8", 8, 3072, 2560),
    ("m12", 12, 3072, 2560),
    ("m16", 16, 3072, 2560),
    ("m20", 20, 3072, 2560),
    ("m24", 24, 3072, 2560),
    ("m28", 28, 3072, 2560),
    ("m32", 32, 3072, 2560),
    ("m36", 36, 3072, 2560),
    ("m40", 40, 3072, 2560),
    ("m44", 44, 3072, 2560),
    ("m48", 48, 3072, 2560),
    ("m56", 56, 3072, 2560),
    ("m64", 64, 3072, 2560),
    ("m72", 72, 3072, 2560),
    ("m80", 80, 3072, 2560),
    ("m88", 88, 3072, 2560),
    ("m96", 96, 3072, 2560),
]

# 非对齐 M（per-rank M 非 16 对齐，k=3072, n=2560）；用 --cases odd 单独跑
ODD_M_CASES = [
    ("m5", 5, 3072, 2560),
    ("m13", 13, 3072, 2560),
    ("m21", 21, 3072, 2560),
    ("m29", 29, 3072, 2560),
    ("m37", 37, 3072, 2560),
    ("m45", 45, 3072, 2560),
    ("m53", 53, 3072, 2560),
    ("m61", 61, 3072, 2560),
    ("m69", 69, 3072, 2560),
    ("m77", 77, 3072, 2560),
    ("m85", 85, 3072, 2560),
    ("m93", 93, 3072, 2560),
]

torch_npu.npu.config.allow_internal_format = True


# ----------------------------------------------------------------------------- 参数
def parse_args():
    parser = argparse.ArgumentParser(
        description="AllGatherMatmulV3 (ubmem) standalone precision/performance test"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="precision",
        choices=["precision", "perf"],
        help="precision=精度比对(CPU golden); perf=性能采集(掩盖率)",
    )
    parser.add_argument(
        "--cases", type=str, default="all", help="逗号分隔的内置用例名, 或 all"
    )
    parser.add_argument(
        "--m", type=int, default=None, help="自定义用例: per-rank M (与 --cases 互斥)"
    )
    parser.add_argument("--k", type=int, default=3072, help="自定义用例: K")
    parser.add_argument("--n", type=int, default=2560, help="自定义用例: N")
    parser.add_argument(
        "--world-size", type=int, default=4, help="rank 数(每 rank 占一个可见设备)"
    )
    parser.add_argument(
        "--port", type=int, default=MASTER_PORT, help="进程组 tcp store 端口"
    )
    parser.add_argument("--rtol", type=float, default=ERROR_TOL, help="精度比对 rtol")
    parser.add_argument("--atol", type=float, default=ERROR_TOL, help="精度比对 atol")
    parser.add_argument(
        "--iters",
        type=int,
        default=10,
        help="精度模式: 融合算子连续调用轮数, 每轮都与 golden 比对(默认 10)",
    )
    parser.add_argument(
        "--golden-iters",
        type=int,
        default=10,
        help="精度模式: golden 小算子级联预跑轮数, 先用脏内置 buffer 再测融合算子(0 关闭)",
    )
    parser.add_argument(
        "--x1-dtype",
        choices=list(QUANT_KINDS),
        default="e4m3",
        help="x1/A 量化类型: e4m3/e5m2/fp4(打包 uint8)",
    )
    parser.add_argument(
        "--x2-dtype",
        choices=list(QUANT_KINDS),
        default="e4m3",
        help="x2/B 量化类型: e4m3/e5m2/fp4(打包 uint8), 与 x1 同为 fp4 或同为 fp8",
    )
    parser.add_argument(
        "--y-dtype",
        choices=list(OUT_KINDS),
        default="bf16",
        help="y 输出类型: bf16/fp16",
    )
    parser.add_argument(
        "--x2-format",
        type=str.lower,
        choices=["nd", "nz"],
        default="nd",
        help="x2 格式(大小写不敏感): nd=普通存储; nz=FRACTAL_NZ 分块(仅 fp8)",
    )
    parser.add_argument(
        "--x2-orient",
        choices=["trans", "non-trans"],
        default="trans",
        help="x2 方向: trans=[N,K] 转置 B(ND 传 .t() 视图); non-trans=[K,N] 非转置 B",
    )
    parser.add_argument("--list-cases", action="store_true", help="列出内置用例后退出")
    return parser.parse_args()


def build_cases(args):
    if args.list_cases:
        for name, m, k, n in DEFAULT_CASES:  # 展示用例列表
            print(f"{name:10s} m={m:4d} k={k} n={n}")
        print("--- odd M ---")
        for name, m, k, n in ODD_M_CASES:
            print(f"{name:10s} m={m:4d} k={k} n={n}")
        sys.exit(0)
    if args.m is not None:  # 自定义形状 m n k组一个用例
        return [(f"custom_m{args.m}_k{args.k}_n{args.n}", args.m, args.k, args.n)]
    table = {name: (name, m, k, n) for name, m, k, n in DEFAULT_CASES + ODD_M_CASES}
    cases = []
    for token in [c.strip() for c in args.cases.split(",") if c.strip()]:
        if token == "all":
            cases.extend(DEFAULT_CASES)
        elif token == "odd":
            cases.extend(ODD_M_CASES)
        elif token in table:
            cases.append(table[token])
        else:
            raise SystemExit(
                f"[ERROR] unknown case: {token}, use --list-cases to view (all / odd / <name>)"
            )
    return cases


# ----------------------------------------------------------------------------- 输入与 golden
def np_fp8_to_torch(np_arr, torch_dtype):
    return torch.from_numpy(np_arr.view(np.uint8)).view(torch_dtype)


def make_quant_data(kind, shape, small):
    """生成 (逻辑数据, 存储字节): fp8 为 1B/元素; fp4 为 2 元素/字节(低 nibble 在前).

    small=True 用于 fp16 输出: 缩小数据/scale 取值, 避免 K 维累加结果溢出 fp16。
    """
    np_dtype = QUANT_KINDS[kind][0]
    lo, hi = (0.5, 2.0) if small else (1.0, 8.0)
    if kind == "fp4":
        logical = np.random.uniform(lo, hi, shape).astype(np_dtype)
        codes = logical.view(np.uint8)
        packed = (codes[..., 0::2] & 0xF) | ((codes[..., 1::2] & 0xF) << 4)
        return logical, packed.astype(np.uint8)
    logical = np.random.uniform(lo, hi, shape).astype(np_dtype)
    return logical, logical.view(np.uint8)


def make_scale(shape, small):
    hi = 2.0 if small else 8.0
    return np.random.uniform(1.0, hi, shape).astype(np_fp8_e8m0fnu)


def quant_data_to_torch(kind, data_bytes):
    """存储字节 -> NPU tensor; fp4 保持 uint8 打包形态, 由 dtype 枚举 296 标识."""
    tensor = torch.from_numpy(np.ascontiguousarray(data_bytes)).npu()
    torch_dtype = QUANT_KINDS[kind][1]
    return tensor if kind == "fp4" else tensor.view(torch_dtype)


def quant_data_to_torch_nz(kind, data_bytes):
    """[N,K] 或 [K,N] 存储字节 -> FRACTAL_NZ 格式 NPU tensor(npu_format_cast 29).

    torch_npu.npu_format_cast 不支持 e5m2 dtype, 因此统一用 uint8 存储,
    由 x2_dtype 枚举(24/23)告知算子按 fp8 解释(bitcast)。
    """
    if kind == "fp4":
        raise ValueError("fp4 does not support FRACTAL_NZ x2")
    tensor = torch.from_numpy(np.ascontiguousarray(data_bytes)).npu()
    return torch_npu.npu_format_cast(tensor, 29)


def fp4_nontrans_to_torch(logical_nk):
    """[N,K] fp4 逻辑数据 -> [K, N/2] uint8(沿 N 打包), 非转置 B."""
    codes = np.ascontiguousarray(logical_nk.T).view(np.uint8)  # [K, N]
    packed = (codes[:, 0::2] & 0xF) | ((codes[:, 1::2] & 0xF) << 4)
    return torch.from_numpy(np.ascontiguousarray(packed)).npu()


def build_x2_tensor(args, logical_nk, data_bytes_nk):
    """按 --x2-format/--x2-orient 构造右矩阵 tensor."""
    if args.x2_orient == "non-trans":
        if args.x2_dtype == "fp4":
            return fp4_nontrans_to_torch(logical_nk)
        bytes_kn = np.ascontiguousarray(data_bytes_nk.T)
        if args.x2_format == "nz":
            return quant_data_to_torch_nz(args.x2_dtype, bytes_kn)
        return quant_data_to_torch(args.x2_dtype, bytes_kn)
    if args.x2_format == "nz":
        return quant_data_to_torch_nz(args.x2_dtype, data_bytes_nk)
    return quant_data_to_torch(args.x2_dtype, data_bytes_nk).t()


def build_x2_scale(args, scale_nk):
    """x2_scale: 转置 B [N, kd, 2]; 非转置 B [kd, N, 2](连续)."""
    if args.x2_orient == "non-trans":
        scale_nk = np.ascontiguousarray(np.transpose(scale_nk, (1, 0, 2)))
    return np_fp8_to_torch(scale_nk, torch.float8_e8m0fnu).npu()


def build_golden_x2(args, data_bytes_nk, scale_nk):
    """golden 小算子级联的 x2: 固定框架口径 ND [N,K] 的 .t() 视图 + [N,kd,2] scale.

    与被测算子的 x2 format/orient 解耦, 保证不同布局下基线口径一致。
    """
    x2_golden = quant_data_to_torch(args.x2_dtype, data_bytes_nk).t()
    x2_scale_golden_u8 = (
        np_fp8_to_torch(scale_nk, torch.float8_e8m0fnu).npu().view(torch.uint8)
    )
    return x2_golden, x2_scale_golden_u8


def generate_inputs(m, k, n, world_size, x1_kind, x2_kind, y_kind):
    """所有 rank 以相同种子序列生成全部 rank 的输入(数据完全一致, 无需通信即可本地算 golden)。

    x1: [M, K] x1_kind; x1_scale: [M, ceil(K/64), 2] e8m0
    x2: [N, K] x2_kind; x2_scale: [N, ceil(K/64), 2] e8m0
    scale 布局: [ceil(K/64), 2] 中第 (g, j) 个值覆盖第 64*g + 32*j 起 32 个元素(MX 32 分组)。
    y_kind=fp16 时缩小数据/scale 范围, 避免 K 维累加溢出 fp16。
    """
    small = y_kind == "fp16"
    a_list, a_bytes_list, a_scale_list = [], [], []
    b_list, b_bytes_list, b_scale_list = [], [], []
    for r in range(world_size):
        np.random.seed(FIXED_SEED + r)
        a_logic, a_bytes = make_quant_data(x1_kind, (m, k), small)
        b_logic, b_bytes = make_quant_data(x2_kind, (n, k), small)
        a_list.append(a_logic)
        a_bytes_list.append(a_bytes)
        b_list.append(b_logic)
        b_bytes_list.append(b_bytes)
        a_scale_list.append(make_scale((m, math.ceil(k / 64), 2), small))
        b_scale_list.append(make_scale((n, math.ceil(k / 64), 2), small))
    return (
        a_list,
        a_scale_list,
        b_list,
        b_scale_list,
        a_bytes_list,
        b_bytes_list,
    )


def dequant_mx(data_fp8, scale_e8m0, k):
    """e8m0 scale [rows, ceil(K/64), 2] 广播到 [rows, K] 后反量化 -> fp32 [rows, K]。"""
    rows = data_fp8.shape[0]
    scale_flat = scale_e8m0.reshape(rows, -1)  # [rows, ceil(K/64)*2]
    scale_full = np.repeat(scale_flat, MX_GROUP_SIZE, axis=-1)[:, :k]  # [rows, K]
    return data_fp8.astype(np.float32) * scale_full.astype(np.float32)


def compute_golden(inputs, rank, k, world_size, out_dtype):
    """CPU golden: dequant(concat(all A)) @ dequant(B_rank)^T -> out_dtype [ws*M, N]。"""
    a_list, a_scale_list, b_list, b_scale_list, _, _ = inputs
    a_deq_all = np.concatenate(
        [dequant_mx(a_list[r], a_scale_list[r], k) for r in range(world_size)], axis=0
    )
    b_deq_nk = dequant_mx(b_list[rank], b_scale_list[rank], k)
    out = torch.from_numpy(a_deq_all) @ torch.from_numpy(b_deq_nk.T)
    return out.to(out_dtype)


# ----------------------------------------------------------------------------- 算子调用
def run_fused_op(x1, x2_t, x1_scale, x2_scale, group, x1_dtype, x2_dtype, y_dtype):
    """调用 all_gather_quant_matmul(ubmem), 返回 y [ws*M, N]。x2 传 [N,K] 物理存的 .t() 视图。"""
    y, _, _ = cann_ops_transformer.all_gather_quant_matmul(
        x1,
        x2_t,
        group,
        x1_scale=x1_scale,
        x2_scale=x2_scale,
        x1_dtype=x1_dtype,
        x2_dtype=x2_dtype,
        y_dtype=y_dtype,
        group_sizes=list(MX_GROUP_SIZES),
        comm_mode="ubmem",
    )
    return y


def run_golden_ops(
    x1, x2_t, x1_scale_u8, x2_scale_u8, group, gather_out, scale_gather_out
):
    """golden 小算子级联: dist._all_gather_base + torch_npu.npu_quant_matmul (与内部框架口径一致)。

    注意: HCCL 集合通信不支持原生 float8_e8m0fnu, e8m0 scale 需以 uint8 视图参与 gather,
    再以 uint8 存储 + 293 枚举传给 npu_quant_matmul(bitcast 语义)。
    """
    dist._all_gather_base(gather_out, x1)
    dist._all_gather_base(scale_gather_out, x1_scale_u8)
    out = torch_npu.npu_quant_matmul(
        x1=gather_out,
        x2=x2_t,
        pertoken_scale=scale_gather_out,
        # x2 传 .t() 视图([K,N])时, x2 的 scale 需配套转置为 [ceil(K/64), N, 2]
        scale=x2_scale_u8.transpose(0, 1),
        offset=None,
        bias=None,
        output_dtype=ACL_BF16,
        x1_dtype=ACL_FP8_E4M3FN,
        x2_dtype=ACL_FP8_E4M3FN,
        pertoken_scale_dtype=ACL_FP8_E8M0,
        scale_dtype=ACL_FP8_E8M0,
        group_sizes=list(MX_GROUP_SIZES),
        y_scale=None,
    )
    return out


# ----------------------------------------------------------------------------- 性能采集
def safe_case_name(name):
    """profiler 目录名: 替换路径分隔符等 shell 敏感字符(case tag 含 -> 和 /)."""
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", name)


def profiled_run(case_name, phase, fn, rank):
    """profiler 内跑 PERF_ITERS 次; i==1 时注入重型算子防 host bound(与内部框架一致)。"""
    prefix = "mc2_" if phase == "mc2" else "hccl_mm_"
    profiling_result = f"./Profiling_Result_{prefix}{safe_case_name(case_name)}"
    if rank == 0:
        shutil.rmtree(profiling_result, ignore_errors=True)
    dist.barrier()
    experimental_config = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        l2_cache=False,
        data_simplification=False,
    )
    heavy = torch.ones(10000, 10000).to(torch.float16).npu()  # 防止 host bound 现象
    with torch_npu.profiler.profile(
        schedule=torch_npu.profiler.schedule(
            wait=0, warmup=0, active=1, repeat=1, skip_first=0
        ),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(profiling_result),
        profile_memory=False,
        with_stack=False,
        with_flops=False,
        with_modules=False,
        experimental_config=experimental_config,
    ):
        for i in range(PERF_ITERS):
            if i == 1:
                for _ in range(50):
                    torch.exp(heavy)
            fn()
    torch.npu.synchronize()
    dist.barrier()
    return profiling_result


def get_avg_perf(all_list, run_time=PERF_ITERS):
    """取后一半迭代的平均(剔除前半段首调抖动), 与内部框架 get_avg_perf 口径一致。"""
    if not all_list:
        return 0.0
    valid_run_time = math.ceil(run_time / 2)
    neg_start_idx = len(all_list) // run_time * valid_run_time
    if neg_start_idx == 0:
        neg_start_idx = len(all_list)
    return sum(all_list[-neg_start_idx:]) / valid_run_time


def parse_op_summary(case_name, phase):
    """解析 Profiling_Result_<phase><case> 下全部 op_summary_*.csv (跨 rank 取最大)。

    golden(hccl_mm_): 分类统计 hccl / matmul / aicpu(HcclLaunchAicpuKernel) / trans(Transpose)
    mc2:             全部非 AI_CPU 行 (融合算子整核耗时)
    返回 dict: mc2 -> perf; golden -> perf, avg_hccl, avg_matmul, avg_aicpu, avg_trans
    """
    prefix = "mc2_" if phase == "mc2" else "hccl_mm_"
    case_path = f"./Profiling_Result_{prefix}{safe_case_name(case_name)}"
    performance = 0.0
    golden_hccl = golden_mm = golden_aicpu = golden_trans = 0.0
    for dirpath, _, filenames in os.walk(case_path):
        for filename in filenames:
            if not filename.startswith("op_summary"):
                continue
            with open(os.path.join(dirpath, filename)) as f:
                csvreader = csv.reader(f)
                headers = next(csvreader)
                t = headers.index("Task Duration(us)")
                task_type = headers.index("Task Type")
                op_type = headers.index("OP Type")
                hccl_list, matmul_list, mc2_list, aicpu_list, trans_list = (
                    [],
                    [],
                    [],
                    [],
                    [],
                )
                for row in csvreader:
                    if not row or row[op_type] == "Exp":  # 跳过注入的重型算子
                        continue
                    if phase == "golden":
                        if row[task_type] in ["HCCL", "COMMUNICATION"]:
                            hccl_list.append(float(row[t]))
                        elif (
                            row[task_type] == "AI_CPU"
                            and row[op_type] == "HcclLaunchAicpuKernel"
                        ):
                            aicpu_list.append(float(row[t]))
                        elif row[op_type] == "Transpose":
                            trans_list.append(float(row[t]))
                        elif row[task_type] in [
                            "AI_VECTOR_CORE",
                            "AI_CORE",
                            "MIX_AIC",
                            "MIX_AIV",
                        ]:
                            matmul_list.append(float(row[t]))
                    else:
                        if row[task_type] != "AI_CPU":
                            mc2_list.append(float(row[t]))
            if phase == "golden":
                avg_hccl = get_avg_perf(hccl_list)
                avg_matmul = get_avg_perf(matmul_list)
                if avg_hccl + avg_matmul > performance:
                    performance = avg_hccl + avg_matmul
                    golden_hccl, golden_mm = avg_hccl, avg_matmul
                    golden_aicpu, golden_trans = (
                        get_avg_perf(aicpu_list),
                        get_avg_perf(trans_list),
                    )
            else:
                avg_mc2 = get_avg_perf(mc2_list)
                if avg_mc2 > performance:
                    performance = avg_mc2
    if phase == "golden":
        return {
            "perf": performance,
            "avg_hccl": golden_hccl,
            "avg_matmul": golden_mm,
            "avg_aicpu": golden_aicpu,
            "avg_trans": golden_trans,
        }
    return {"perf": performance}


def append_csv(path, title, row):
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(title)
        writer.writerow(row)


# ----------------------------------------------------------------------------- worker
def precision_flow(rank, case, args, group):
    name, m, k, n = case
    inputs = generate_inputs(
        m, k, n, args.world_size, args.x1_dtype, args.x2_dtype, args.y_dtype
    )
    a_list, a_scale_list, b_list, b_scale_list, a_bytes_list, b_bytes_list = inputs

    x1 = quant_data_to_torch(args.x1_dtype, a_bytes_list[rank])
    x2_t = build_x2_tensor(args, b_list[rank], b_bytes_list[rank])
    x1_scale = np_fp8_to_torch(a_scale_list[rank], torch.float8_e8m0fnu).npu()
    x2_scale = build_x2_scale(args, b_scale_list[rank])

    x1_acl_dtype = QUANT_KINDS[args.x1_dtype][2]
    x2_acl_dtype = QUANT_KINDS[args.x2_dtype][2]
    y_dtype, y_acl_dtype = OUT_KINDS[args.y_dtype]

    expected = (m * args.world_size, n)

    # golden 小算子级联预跑: 仅 e4m3 x e4m3 -> bf16 组合支持(其余组合直接测融合算子)
    golden_ops_ok = (
        args.x1_dtype == "e4m3" and args.x2_dtype == "e4m3" and args.y_dtype == "bf16"
    )
    # 先把buffer用脏，后续融合算子再使用
    if args.golden_iters > 0 and golden_ops_ok:
        x1_scale_u8 = x1_scale.view(torch.uint8)
        x2_golden, x2_scale_golden_u8 = build_golden_x2(
            args, b_bytes_list[rank], b_scale_list[rank]
        )
        gather_out = torch.zeros(
            (args.world_size * m, k), dtype=torch.float8_e4m3fn
        ).npu()
        scale_gather_out = torch.zeros(
            (args.world_size * m, math.ceil(k / 64), 2), dtype=torch.uint8
        ).npu()
        for _ in range(args.golden_iters):
            run_golden_ops(
                x1,
                x2_golden,
                x1_scale_u8,
                x2_scale_golden_u8,
                group,
                gather_out,
                scale_gather_out,
            )
        torch.npu.synchronize()
    elif args.golden_iters > 0:
        print(
            f"[rank {rank}] skip golden pre-run: "
            f"{args.x1_dtype} x {args.x2_dtype} -> {args.y_dtype}",
            flush=True,
        )

    golden = compute_golden(inputs, rank, k, args.world_size, y_dtype)

    all_pass = True
    for i in range(args.iters):
        y = run_fused_op(
            x1, x2_t, x1_scale, x2_scale, group, x1_acl_dtype, x2_acl_dtype, y_acl_dtype
        )
        torch.npu.synchronize()
        y_cpu = y.cpu()
        assert tuple(y.shape) == expected, f"y shape {tuple(y.shape)} != {expected}"
        assert y.dtype == y_dtype, f"y dtype {y.dtype} != {y_dtype}"
        # 由 allclose(equal_nan=True) 判定(同号 inf 也判等)。仅在出现非有限值时告警,
        y_finite = torch.isfinite(y_cpu.float())
        g_finite = torch.isfinite(golden.float())
        if not (y_finite.all() and g_finite.all()):
            print(
                f"[rank {rank}] {name} iter{i + 1}/{args.iters}: WARN non-finite elements "
                f"(y={int((~y_finite).sum())}, golden={int((~g_finite).sum())}), "
                f"matched by allclose(equal_nan=True); case may be vacuous.",
                flush=True,
            )
        abs_err = (y_cpu.float() - golden.float()).abs()
        passed = torch.allclose(
            golden, y_cpu, rtol=args.rtol, atol=args.atol, equal_nan=True
        )
        all_pass = all_pass and passed
        print(
            f"[rank {rank}] {name} iter{i + 1}/{args.iters}: abs_err max={abs_err.max().item():.6e}, "
            f"over_tol={(abs_err > args.atol).sum().item()}/{expected[0] * expected[1]}, "
            f"result={'PASS' if passed else 'FAIL'}",
            flush=True,
        )
    return name, rank, "PASS" if all_pass else "FAIL"


def perf_flow(rank, case, args, group):
    """性能: golden 小算子级联 + mc2 融合算子各采 10 次, rank0 解析并输出掩盖率。"""
    name, m, k, n = case
    inputs = generate_inputs(
        m, k, n, args.world_size, args.x1_dtype, args.x2_dtype, args.y_dtype
    )
    a_list, a_scale_list, b_list, b_scale_list, a_bytes_list, b_bytes_list = inputs

    x1 = quant_data_to_torch(args.x1_dtype, a_bytes_list[rank])
    x2_t = build_x2_tensor(args, b_list[rank], b_bytes_list[rank])
    x1_scale = np_fp8_to_torch(a_scale_list[rank], torch.float8_e8m0fnu).npu()
    x2_scale = build_x2_scale(args, b_scale_list[rank])

    ws = args.world_size
    x1_scale_u8 = x1_scale.view(torch.uint8)  # e8m0 以 uint8 视图参与 HCCL gather
    # golden 基线固定按框架口径构造, 与被测算子的 x2 布局(ND/NZ x 转置/非转置)解耦
    x2_golden, x2_scale_golden_u8 = build_golden_x2(
        args, b_bytes_list[rank], b_scale_list[rank]
    )
    gather_out = torch.zeros((ws * m, k), dtype=torch.float8_e4m3fn).npu()
    scale_gather_out = torch.zeros(
        (ws * m, math.ceil(k / 64), 2), dtype=torch.uint8
    ).npu()

    x1_acl_dtype = QUANT_KINDS[args.x1_dtype][2]
    x2_acl_dtype = QUANT_KINDS[args.x2_dtype][2]
    y_dtype, y_acl_dtype = OUT_KINDS[args.y_dtype]

    # 预热一次（建链/JIT/首调），不计入采集；保留首调输出用于首轮精度校验
    y_first = run_fused_op(
        x1, x2_t, x1_scale, x2_scale, group, x1_acl_dtype, x2_acl_dtype, y_acl_dtype
    )
    run_golden_ops(
        x1,
        x2_golden,
        x1_scale_u8,
        x2_scale_golden_u8,
        group,
        gather_out,
        scale_gather_out,
    )
    torch.npu.synchronize()
    dist.barrier()

    profiled_run(
        name,
        "golden",
        lambda: run_golden_ops(
            x1,
            x2_golden,
            x1_scale_u8,
            x2_scale_golden_u8,
            group,
            gather_out,
            scale_gather_out,
        ),
        rank,
    )
    profiled_run(
        name,
        "mc2",
        lambda: run_fused_op(
            x1, x2_t, x1_scale, x2_scale, group, x1_acl_dtype, x2_acl_dtype, y_acl_dtype
        ),
        rank,
    )

    # 首轮 + 末轮精度校验: warmup 首调输出 + 性能跑完后的末调输出, 两端均 PASS 才算 PASS,
    y_final = run_fused_op(
        x1, x2_t, x1_scale, x2_scale, group, x1_acl_dtype, x2_acl_dtype, y_acl_dtype
    )
    torch.npu.synchronize()
    golden = compute_golden(inputs, rank, k, ws, y_dtype)
    y_first_cpu = y_first.cpu()
    y_final_cpu = y_final.cpu()

    def _prec_ok(y_cpu):
        return (
            torch.isfinite(y_cpu.float()).all()
            and torch.isfinite(golden.float()).all()
            and torch.allclose(
                golden, y_cpu, rtol=args.rtol, atol=args.atol, equal_nan=True
            )
        )

    prec_first, prec_final = _prec_ok(y_first_cpu), _prec_ok(y_final_cpu)
    ok_flag = torch.tensor(
        [1 if prec_first else 0, 1 if prec_final else 0], dtype=torch.int32
    ).npu()
    dist.all_reduce(ok_flag, op=dist.ReduceOp.MIN)
    precision_first = "PASS" if ok_flag[0].item() == 1 else "FAIL"
    precision_last = "PASS" if ok_flag[1].item() == 1 else "FAIL"
    precision_all_ranks = (
        "PASS" if (precision_first == "PASS" and precision_last == "PASS") else "FAIL"
    )
    # 用rank 0 为代表进行统计
    if rank == 0:
        golden_stat = parse_op_summary(name, "golden")
        mc2 = parse_op_summary(name, "mc2")
        mc2_prof, golden_prof = mc2["perf"], golden_stat["perf"]
        avg_hccl, avg_matmul = golden_stat["avg_hccl"], golden_stat["avg_matmul"]
        avg_aicpu, avg_trans = golden_stat["avg_aicpu"], golden_stat["avg_trans"]
        if avg_hccl and avg_matmul and mc2_prof:
            mask = (golden_prof - mc2_prof) / min(avg_hccl, avg_matmul + avg_trans)
            result = "PASS" if mask > 0.7 else "FAIL"
        else:
            mask, result = 0.0, "N/A"
        print(
            f"[perf] {name}: mc2={mc2_prof:.2f}us golden={golden_prof:.2f}us "
            f"(hccl={avg_hccl:.2f} mm={avg_matmul:.2f} aicpu={avg_aicpu:.2f} trans={avg_trans:.2f}) "
            f"mask={mask:.3f} -> {result}, "
            f"precision(first,last, all ranks)=({precision_first},{precision_last})",
            flush=True,
        )
        append_csv(
            "all_gather_matmul_v3_performance_result.csv",
            [
                "case_name",
                "world_size",
                "mc2_prof(us)",
                "golden_prof(us)",
                "avg_hccl",
                "avg_matmul",
                "avg_aicpu",
                "avg_trans",
                "mask_rate",
                "result",
                "precision",
                "timestamp",
            ],
            [
                name,
                ws,
                f"{mc2_prof:.2f}",
                f"{golden_prof:.2f}",
                f"{avg_hccl:.2f}",
                f"{avg_matmul:.2f}",
                f"{avg_aicpu:.2f}",
                f"{avg_trans:.2f}",
                f"{mask:.4f}",
                result,
                precision_all_ranks,
                time.strftime("%Y-%m-%d %H:%M:%S"),
            ],
        )
    dist.barrier()


def worker(queue, rank, args, cases):
    try:
        torch_npu.npu.set_device(rank)
        dist.init_process_group(
            backend="hccl",
            rank=rank,
            world_size=args.world_size,
            init_method=f"tcp://{MASTER_IP}:{args.port}",
        )
        group = dist.new_group(ranks=list(range(args.world_size)))
        print(f"[INFO] device_{rank} HCCL comm group created", flush=True)

        for case in cases:
            if args.mode == "precision":
                result = precision_flow(rank, case, args, group)
                queue.put(["precision", *result])
            else:
                perf_flow(rank, case, args, group)
                queue.put(["perf", case[0], rank, "DONE"])
        dist.barrier()
    except Exception as e:
        import traceback

        traceback.print_exc()
        queue.put([args.mode, "unknown", rank, f"error: {e}"])
    finally:
        if dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:
                pass
        torch.npu.empty_cache()


# ----------------------------------------------------------------------------- main
def main():
    args = parse_args()
    if args.mode == "perf" and (
        args.x1_dtype != "e4m3" or args.x2_dtype != "e4m3" or args.y_dtype != "bf16"
    ):
        raise SystemExit(
            "[ERROR] perf mode supports only e4m3 x e4m3 -> bf16 "
            "(golden small-op cascade is e4m3-only)"
        )
    if (args.x1_dtype == "fp4") != (args.x2_dtype == "fp4"):
        raise SystemExit(
            "[ERROR] fp4 requires both --x1-dtype and --x2-dtype to be fp4"
        )
    if args.x2_format == "nz" and args.x2_dtype == "fp4":
        raise SystemExit("[ERROR] fp4 does not support FRACTAL_NZ x2")
    dtype_tag = f"{args.x1_dtype}x{args.x2_dtype}->{args.y_dtype}/{args.x2_format}/{args.x2_orient}"
    cases = [(f"{name}[{dtype_tag}]", m, k, n) for name, m, k, n in build_cases(args)]
    print(
        f"[INFO] AllGatherMatmulV3(ubmem) test: mode={args.mode}, world_size={args.world_size}, "
        f"dtypes={dtype_tag}, cases={[c[0] for c in cases]}"
    )

    torch.multiprocessing.set_start_method("spawn", force=True)
    manager = Manager()
    queue = manager.Queue()

    procs = []
    for rank in range(args.world_size):
        p = Process(target=worker, args=(queue, rank, args, cases))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

    # 进程已全部退出后再非阻塞收集结果, 避免按条数 get 时 worker 崩溃导致永久阻塞
    results = []
    while True:
        try:
            results.append(queue.get_nowait())
        except Exception:
            break
    if not results:
        print(
            f"\n[RESULT] {args.mode}: no rank results collected (all workers crashed?), see traceback above"
        )
        sys.exit(1)
    if args.mode == "precision":
        title = ["case_name", "rank", "result", "timestamp"]
        rows, all_pass = [], True
        for _, name, rank, res in sorted(results):
            rows.append([name, rank, res, time.strftime("%Y-%m-%d %H:%M:%S")])
            all_pass = all_pass and res == "PASS"
        for row in rows:
            append_csv("all_gather_matmul_v3_precision_result.csv", title, row)
        n_pass = sum(1 for r in rows if r[2] == "PASS")
        print(
            f"\n{'=' * 70}\n[RESULT] precision: {n_pass}/{len(rows)} PASS "
            f"-> {'ALL PASS' if all_pass else 'FAILED'}\n{'=' * 70}"
        )
        sys.exit(0 if all_pass else 1)
    else:
        errors = [r for r in results if len(r) >= 4 and str(r[3]).startswith("error")]
        if errors:
            print(
                f"\n{'=' * 70}\n[RESULT] perf FAILED: {len(errors)} rank error(s), see traceback above\n{'=' * 70}"
            )
            sys.exit(1)
        print(
            f"\n{'=' * 70}\n[RESULT] perf done, see all_gather_matmul_v3_performance_result.csv\n{'=' * 70}"
        )


if __name__ == "__main__":
    main()
