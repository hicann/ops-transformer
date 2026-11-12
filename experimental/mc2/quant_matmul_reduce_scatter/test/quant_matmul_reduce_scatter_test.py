#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""QuantMatmulReduceScatter（ubmem）独立测试：精度模式 + 性能模式。

需要已安装该分支的算子包及 cann_ops_transformer(_custom) Python 扩展。

计算语义（每张来源卡 j）：
  1. 本卡量化矩阵乘：C_j = dequant(A_j) @ dequant(B_j)，形状 [R*M, N]。
  2. ReduceScatter：输出卡 r 取所有 C_j 的第 r 段行求和，得到 [M, N]。
  M 是每卡输出行数（perRankM），R 是 world-size；输入 A 的行数为 R*M。
  A：ND；--x1-dtype/--x2-dtype 选择 e4m3、e5m2、fp4（E2M1）。
  --y-dtype 选择 bf16/fp16；--x2-format 选择 nd/nz；--x2-orient 选择 trans/non-trans。
  默认 e4m3×e4m3→bf16、ND/trans；显式选择 NZ 时默认 non-trans。
  FP4 仅支持两输入同为 FP4、ND；打包轴须为偶数。
  这些是脚本能力；实际执行还要求安装的算子和扩展支持所选组合。
  trans：B 物理存 [N,K] 后传 .t() 视图；non-trans：B 物理存 [K,N]。
  NZ 在转置视图之前执行 format_cast(29)；scale 由方向决定，与 ND/NZ 独立。
  根据 B 的布局推断方向；--x2-orient 仍用于构造 B 和 scale。
  两路 MX scale：uint8 保存的 E8M0，调用时显式传 293。

用法（也可通过同目录 bash run_test.sh 传递相同参数）：
  # 四卡精度：默认 perRankM=4/20/36/48，K=2048、N=3584，每组连续测三轮
  python3 quant_matmul_reduce_scatter_test.py --mode precision --world-size 4
  # 指定内置用例 / 自定义 shape，--m 表示每卡输出 M
  python3 quant_matmul_reduce_scatter_test.py --mode precision --cases b4_mtp4,b20_mtp4
  python3 quant_matmul_reduce_scatter_test.py --mode precision --m 64 --k 2048 --n 3584
  # 普通性能：原 profile 的 50 轮、1 轮 warmup，保留 Max/Exp 干扰负载
  python3 quant_matmul_reduce_scatter_test.py --mode perf --world-size 4
  # ND 精度/性能：
  python3 quant_matmul_reduce_scatter_test.py --mode precision --x2-format nd --m 8 --k 2048 --n 3584
  python3 quant_matmul_reduce_scatter_test.py --mode perf --x2-format nd --with-baseline
  python3 quant_matmul_reduce_scatter_test.py --x1-dtype e5m2 --x2-dtype e4m3 --y-dtype fp16 --x2-format nd --x2-orient trans
  python3 quant_matmul_reduce_scatter_test.py --x1-dtype fp4 --x2-dtype fp4 --x2-format nd --x2-orient non-trans
  # 可选小算子级联对照：QuantMatmul + HCCL ReduceScatter，额外报告掩盖率
  python3 quant_matmul_reduce_scatter_test.py --mode perf --world-size 4 --with-baseline
  python3 quant_matmul_reduce_scatter_test.py --list-cases

环境准备：
  激活安装了 torch/torch_npu/本分支扩展的 Python 环境；source CANN set_env.sh
  及自定义算子包环境。隔离安装时设置相应 PYTHONPATH、ASCEND_CUSTOM_OPP_PATH。
  可见设备由排队系统/ASCEND_RT_VISIBLE_DEVICES 决定，world-size 不应超过设备数。
  默认通信组按 mc2_test 显式配置 CCU。
  HCCL_OP_EXPANSION_MODE 未设置时使用 CCU_SCHED，供未显式配置的通信组使用。

输出与统计口径：
  每次在 --output-dir 下新建独立目录，顶层只保存 config.json 和对应模式的汇总 CSV，逐卡 JSON/NPY 与 Profiler 收进 details/。
  精度：反量化 FP32 matmul → 每卡 BF16 舍入 → FP32 归约 → BF16，连续调用结束后，只检查最后一次输出的所有 rank。
  普通性能：沿用原 profile 的逐轮 prof.step()，统计 active 记录的后半段均值。
  --with-baseline：仅额外采集小算子级联，与融合路径使用相同的采集和统计规则。
        两种模式均排除 Max/Exp 耗时。
  baseline_us=matmul_us+hccl_us；AICPU 启动记录仅作诊断，不参与主耗时校验或计算。
  aicpu_raw_samples 保存记录数；未建立逐轮对应关系时 aicpu_us 留空，状态单独列出。
  prof_rate=baseline_us/mc2_us；mask=(baseline_us-mc2_us)/min(matmul_us,hccl_us)。
        指标仅报告，不设置合格门槛，也不代表完整 Host 调用耗时。
  精度沿用 allclose(equal_nan=True)；精度不符、worker 异常、缺少结果或采集记录异常返回非零。
"""

import argparse
import csv
import json
import math
import inspect
import os
from pathlib import Path
import subprocess
import sys
import time
from importlib import import_module
from itertools import product

# 必须在 HCCL 初始化前设置
os.environ.setdefault("HCCL_BUFFSIZE", "1000")
os.environ.setdefault("HCCL_OP_EXPANSION_MODE", "CCU_SCHED")

import numpy as np
from ml_dtypes import float4_e2m1fn
import torch
import torch_npu
import torch.distributed as dist
import torch.multiprocessing as mp

for package in ("cann_ops_transformer_custom", "cann_ops_transformer"):
    module_name = package + ".ops.mc2.quant_matmul_reduce_scatter"
    try:
        module = import_module(module_name)
        break
    except ModuleNotFoundError as exc:
        # 扩展内部缺少依赖时保留原始错误，不误当成扩展包未安装。
        if exc.name != package and not module_name.startswith(exc.name + "."):
            raise
else:
    raise ModuleNotFoundError(
        "Install the quant_matmul_reduce_scatter torch_extension wheel first"
    )

quant_matmul_reduce_scatter = module.quant_matmul_reduce_scatter
torch_npu.npu.config.allow_internal_format = True

DEFAULT_CASES = [
    # (case_name, m_per_rank, k, n)
    ("b4_mtp4", 4, 2048, 3584),
    ("b20_mtp4", 20, 2048, 3584),
    ("b36_mtp4", 36, 2048, 3584),
    ("b48_mtp4", 48, 2048, 3584),
    ("b4_mtp8", 8, 2048, 3584),
    ("b20_mtp8", 40, 2048, 3584),
    ("b36_mtp8", 72, 2048, 3584),
    ("b48_mtp8", 96, 2048, 3584),
    # 额外精度验证形状：
    ("extra_m64", 64, 2048, 3584),
    ("extra_m128", 128, 2048, 3584),
]
MX_GROUP_SIZE = 32
QUANT_DTYPES = {
    "e4m3": (torch.float8_e4m3fn, 24),
    "e5m2": (torch.float8_e5m2, 23),
    "fp4": (torch.uint8, 296),
}
OUTPUT_DTYPES = {"bf16": (torch.bfloat16, 15), "fp16": (torch.float16, 5)}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("precision", "perf", "all"), default="precision"
    )
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument(
        "--cases", default=None, help="all or comma-separated case names"
    )
    parser.add_argument(
        "--m", type=int, help="OUTPUT M per rank; input M = m * world-size"
    )
    parser.add_argument("--k", type=int, default=2048)
    parser.add_argument("--n", type=int, default=3584)
    parser.add_argument(
        "--x2-format",
        choices=("nd", "nz"),
        default="nd",
        help="B layout: ND (default) or FRACTAL_NZ",
    )
    parser.add_argument("--x1-dtype", choices=QUANT_DTYPES, default="e4m3")
    parser.add_argument("--x2-dtype", choices=QUANT_DTYPES, default="e4m3")
    parser.add_argument("--y-dtype", choices=OUTPUT_DTYPES, default="bf16")
    parser.add_argument(
        "--x2-orient",
        choices=("trans", "non-trans"),
        help="B orientation; defaults to trans for ND and non-trans for NZ",
    )
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--port", type=int, default=50003)
    parser.add_argument("--master-ip", default="127.0.0.1")
    parser.add_argument(
        "--iters", type=int, default=3, help="precision calls, check the last output"
    )
    parser.add_argument("--loop-time", type=int, default=50)
    parser.add_argument(
        "--warmup-time", type=int, default=1, help="ordinary perf profiler warmup"
    )
    parser.add_argument(
        "--exp-count",
        type=int,
        default=100,
        help="Exp calls after iteration 1; 0 disables",
    )
    parser.add_argument(
        "--with-baseline",
        action="store_true",
        help="also profile npu_quant_matmul + reduce_scatter_tensor and report overlap",
    )
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args(argv)
    # Match mc2_test get_hcomm_info(): configure the default group used by RS.
    args.hccl_config = {"hccl_op_expansion_mode": 6}
    if args.x2_orient is None:
        args.x2_orient = "trans" if args.x2_format == "nd" else "non-trans"
    if (args.x1_dtype == "fp4") != (args.x2_dtype == "fp4"):
        parser.error("FP4 requires both inputs to be fp4")
    if args.x2_dtype == "fp4" and args.x2_format == "nz":
        parser.error("FP4 NZ is not supported in reference PR 15052")
    if args.list_cases:
        return args
    if args.m is not None and args.cases is not None:
        parser.error("--m and --cases are mutually exclusive")
    if min(args.world_size, args.k, args.n, args.iters, args.loop_time) <= 0:
        parser.error("world-size, K, N, iters and loop-time must be positive")
    if args.m is not None and args.m <= 0:
        parser.error("M must be positive")
    if args.exp_count < 0:
        parser.error("exp-count must be nonnegative")
    if not 0 <= args.warmup_time < args.loop_time:
        parser.error("require 0 <= warmup-time < loop-time")
    if (
        not 0 < args.port < 65536
        or not math.isfinite(args.rtol)
        or not math.isfinite(args.atol)
        or args.rtol < 0
        or args.atol < 0
    ):
        parser.error("invalid port or tolerance")
    if args.with_baseline and args.mode == "precision":
        parser.error("--with-baseline requires --mode perf or all")
    return args


def build_cases(args):
    if args.m is not None:
        return [(f"custom_m{args.m}_k{args.k}_n{args.n}", args.m, args.k, args.n)]
    names = args.cases
    if names is None:
        names = (
            "b4_mtp8,b20_mtp8,b36_mtp8,b48_mtp8"
            if args.world_size == 8
            else "b4_mtp4,b20_mtp4,b36_mtp4,b48_mtp4"
        )
    if names == "all":
        return list(DEFAULT_CASES)
    table = {case[0]: case for case in DEFAULT_CASES}
    wanted = [name.strip() for name in names.split(",") if name.strip()]
    if (
        not wanted
        or len(set(wanted)) != len(wanted)
        or any(name not in table for name in wanted)
    ):
        raise ValueError("unknown, empty or duplicate cases; use --list-cases")
    return [table[name] for name in wanted]


# ------------------------------ 数据生成与 CPU 参考 ------------------------------
def broadcast_mx_scale(scale, axis, k, n):
    """E8M0 字节 b 表示 2**(b-127)；每个 scale 覆盖 K 方向 32 个元素。

    A scale：[R*M, ceil(K/64), 2]；转置 B：[N,kd,2]，非转置 B：[kd,N,2]。
    最后一维的两个 scale 分别对应同一 64 元素组的前/后 32 个元素。
    """
    sf = torch.pow(torch.tensor(2.0), scale.to(torch.float32) - 127.0)
    sf = sf.repeat_interleave(MX_GROUP_SIZE, dim=-1)
    if axis == 1:
        return sf.reshape(sf.shape[0], -1)[:, :k]
    return sf.permute(0, 2, 1).reshape(-1, n)[:k, :]


def quantize_cpu(data, kind):
    if kind == "fp4":
        return torch.from_numpy(data.numpy().astype(float4_e2m1fn).astype(np.float32))
    return data.to(QUANT_DTYPES[kind][0])


def to_npu_quant(data, kind, nz=False, transpose=False):
    """参照 PR 15052：FP4 两元素/字节，低 nibble 在前；NZ 使用字节存储。"""
    if kind == "fp4":
        if data.shape[-1] % 2:
            raise ValueError("FP4 packed axis must be even")
        codes = data.float().numpy().astype(float4_e2m1fn).view(np.uint8)
        packed = (codes[..., 0::2] & 15) | ((codes[..., 1::2] & 15) << 4)
        result = torch.from_numpy(np.ascontiguousarray(packed)).npu()
    elif nz:
        # E4M3 继续使用原生 tensor，保留当前 RS 扩展的兼容入口。
        storage = data if kind == "e4m3" else data.view(torch.uint8)
        result = torch_npu.npu_format_cast(storage.npu(), 29)
    else:
        result = data.npu()
    return result.t() if transpose else result


def build_b(data, args):
    return to_npu_quant(
        data,
        args.x2_dtype,
        args.x2_format == "nz",
        args.x2_format == "nd" and args.x2_orient == "trans",
    )


def compute_golden(
    x1_list,
    x2_nd_list,
    x1_scale_list,
    x2_scale_list,
    m_per_rank,
    k,
    n,
    world_size,
    x2_format="nz",
    x2_orient=None,
    y_dtype="bf16",
):
    """Matmul 和 RS 均按所选输出类型舍入，RS 在 FP32 中累加。"""
    trans = (x2_orient or ("trans" if x2_format == "nd" else "non-trans")) == "trans"
    out_dtype = OUTPUT_DTYPES[y_dtype][0]
    partials = []
    for r in range(world_size):
        a = x1_list[r].float() * broadcast_mx_scale(x1_scale_list[r], 1, k, n)
        b = x2_nd_list[r].float() * broadcast_mx_scale(
            x2_scale_list[r], 1 if trans else 0, k, n
        )
        partials.append((a @ (b.t() if trans else b)).to(out_dtype))
    result = []
    for r in range(world_size):
        acc = torch.zeros(m_per_rank, n)
        for partial in partials:
            acc += partial[r * m_per_rank : (r + 1) * m_per_rank].float()
        value = acc.to(out_dtype).float()
        if not torch.isfinite(value).all():
            raise ValueError(
                "Golden contains non-finite values; reduce input/scale range"
            )
        result.append(value)
    return result


def prepare_precision_case(case, world_size, args):
    _, m, k, n = case
    total_m, kd = m * world_size, math.ceil(k / 64)
    trans = args.x2_orient == "trans"
    a_list, b_list, sa_list, sb_list = [], [], [], []
    for rank in range(world_size):
        np.random.seed(42 + rank * 1000)
        # FP16 输入范围还需计入 RS 的卡数，避免 Inf 对 Inf 被误判通过。
        hi = (
            min(2.0, math.sqrt(30000.0 / (k * world_size)))
            if args.y_dtype == "fp16"
            else 8.0
        )
        lo = min(0.5, hi / 2) if args.y_dtype == "fp16" else 1.0
        a = torch.from_numpy(np.random.uniform(lo, hi, (total_m, k)).astype(np.float32))
        b = torch.from_numpy(
            np.random.uniform(lo, hi, (n, k) if trans else (k, n)).astype(np.float32)
        )
        a_list.append(quantize_cpu(a, args.x1_dtype))
        b_list.append(quantize_cpu(b, args.x2_dtype))
        scale_hi = 128 if args.y_dtype == "fp16" else 131
        sb_list.append(
            torch.from_numpy(
                np.random.randint(
                    126, scale_hi, (n, kd, 2) if trans else (kd, n, 2), dtype=np.uint8
                )
            )
        )
        sa_list.append(
            torch.from_numpy(
                np.random.randint(126, scale_hi, (total_m, kd, 2), dtype=np.uint8)
            )
        )
    golden = compute_golden(
        a_list,
        b_list,
        sa_list,
        sb_list,
        m,
        k,
        n,
        world_size,
        args.x2_format,
        args.x2_orient,
        args.y_dtype,
    )
    return list(zip(a_list, b_list, sa_list, sb_list)), golden


def fused(inputs, group, args):
    kwargs = dict(comm_mode="ubmem", x1_scale_dtype=293, x2_scale_dtype=293)
    for index, kind in enumerate((args.x1_dtype, args.x2_dtype)):
        if kind != "e4m3" or inputs[index].dtype == torch.uint8:
            kwargs[f"x{index + 1}_dtype"] = {"e4m3": 292, "e5m2": 291, "fp4": 296}[kind]
    if "y_dtype" in inspect.signature(quant_matmul_reduce_scatter).parameters:
        kwargs["y_dtype"] = {"bf16": 283, "fp16": 257}[args.y_dtype]
    elif args.y_dtype != "bf16":
        raise RuntimeError(
            "Installed extension lacks y_dtype; FP16 requires updated operator/extension"
        )
    return quant_matmul_reduce_scatter(*inputs, group, **kwargs)


def baseline_config(args):
    if args.x2_format == "nz" and "e5m2" in (args.x1_dtype, args.x2_dtype):
        return argparse.Namespace(
            **(vars(args) | {"x1_dtype": "e4m3", "x2_dtype": "e4m3"})
        )
    return args


def prepare_baseline_inputs(inputs, args):
    a, b, sa, sb = inputs
    if args.x2_orient == "trans":
        # 与 B 的 .t() 配套保留转置 stride；contiguous() 会使小算子判定二者转置状态不一致。
        sb = sb.transpose(0, 1)
    return a, b, sa, sb


def baseline(inputs, group, output, args):
    a, b, sa, sb = inputs
    c = torch_npu.npu_quant_matmul(
        x1=a,
        x2=b,
        pertoken_scale=sa,
        scale=sb,
        x1_dtype=QUANT_DTYPES[args.x1_dtype][1],
        x2_dtype=QUANT_DTYPES[args.x2_dtype][1],
        output_dtype=OUTPUT_DTYPES[args.y_dtype][1],
        pertoken_scale_dtype=293,
        scale_dtype=293,
        group_sizes=[1, 1, 32],
    )
    dist.reduce_scatter_tensor(output, c, op=dist.ReduceOp.SUM)
    return output


def compare(output, golden, args):
    actual = output.cpu().float()
    if tuple(actual.shape) != tuple(golden.shape):
        raise AssertionError(
            f"unexpected output: shape={output.shape}, dtype={output.dtype}"
        )
    passed = torch.allclose(
        actual, golden, rtol=args.rtol, atol=args.atol, equal_nan=True
    )
    abs_error = (actual - golden).abs()
    close = (
        torch.isclose(actual, golden, rtol=args.rtol, atol=args.atol, equal_nan=True)
        .sum()
        .item()
    )
    worst = abs_error.argmax().item()
    return (
        bool(passed),
        abs_error.flatten()[worst].item(),
        close,
        golden.numel(),
        actual.flatten()[worst].item(),
        golden.flatten()[worst].item(),
    )


# ------------------------------ 性能采集与设备耗时解析 ------------------------------
def profile_calls(fn, path, args, cache_input, heavy, output_state):
    """连续提交调用，结束后同步；不在每轮插入同步，以保留原性能负载。

    第 2 轮后插入 Exp 负载，沿用原 profile 脚本用于缓解 Host 提交不足的做法。
    变量 cache_input 只有 49152 个 FP32 元素，不能按旧变量名理解为 192MB。
    """
    # 两条路径共用原 profile 的采集配置；是否测基线不影响采集规则。
    config = torch_npu.profiler._ExperimentalConfig(
        export_type=torch_npu.profiler.ExportType.Text,
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        l2_cache=False,
        data_simplification=False,
    )
    with torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        schedule=torch_npu.profiler.schedule(
            wait=0,
            warmup=args.warmup_time,
            active=args.loop_time - args.warmup_time,
            repeat=1,
            skip_first=0,
        ),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(path)),
        profile_memory=True,
        with_stack=False,
        with_flops=False,
        with_modules=False,
        experimental_config=config,
    ) as prof:
        for iteration in range(args.loop_time):
            scale_l2 = torch.max(cache_input)
            output_state[0] = fn()
            if iteration == 1:
                for _ in range(args.exp_count):
                    torch.exp(heavy)
            prof.step()
    torch.npu.synchronize()


def parse_summary(path, phase, *, expected_samples=49, warmup_samples=1):
    """读取当前 case/rank/phase 的原始 op_summary，提取目标设备任务。

    两条路径均取 active 记录的后半段均值，baseline 开关不改变统计规则。
    当前串行路径每轮一条目标任务；末尾窗口须连续，早期缺失另需完整卡证明。
    """

    def read_records(filename):
        values = {"fused": [], "matmul": [], "hccl": [], "aicpu": []}
        with filename.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                op = row.get("OP Type", "").lower().replace("_", "")
                task = row.get("Task Type", "").upper()
                category = None
                if (
                    phase == "mc2"
                    and "quantmatmulreducescatter" in op
                    and task != "AI_CPU"
                ):
                    category = "fused"
                elif phase == "baseline":
                    if task == "AI_CPU" and op == "hccllaunchaicpukernel":
                        category = "aicpu"
                    elif task != "AI_CPU" and "reducescatter" in op:
                        category = "hccl"
                    elif "matmul" in op and task != "AI_CPU":
                        category = "matmul"
                if category:
                    values[category].append(
                        (
                            float(row["Task Start Time(us)"]),
                            float(row["Task Duration(us)"]),
                            row.get("Task ID", ""),
                            row.get("Op Name", ""),
                        )
                    )
        return values

    streams = []
    for filename in sorted(path.rglob("op_summary*.csv")):
        values = read_records(filename)
        streams.append(values)
    required = ("fused",) if phase == "mc2" else ("matmul", "hccl")
    valid = [s for s in streams if all(s[key] for key in required)]
    if not valid:
        raise RuntimeError(
            f"No complete {phase} op_summary found under {path}; inspect raw profiler data"
        )
    if len(valid) != 1:
        raise RuntimeError(
            f"Expected one device op_summary under {path}, found {len(valid)}"
        )
    result = {}
    for key in required:
        records = sorted(valid[0][key])
        if any(
            not math.isfinite(r[0]) or not math.isfinite(r[1]) or r[1] <= 0
            for r in records
        ):
            raise RuntimeError(f"Invalid device timing for {phase}/{key}")
        used_samples = math.ceil(expected_samples / 2)
        if not used_samples <= len(records) <= expected_samples + warmup_samples:
            raise RuntimeError(
                f"{phase}/{key}: insufficient or duplicate records: {len(records)}"
            )
        selected = records[-used_samples:]
        # 采集开始时可漏掉早期任务；不足 active 条时，必须有同组完整卡证明末尾窗口一致。
        if len(records) < expected_samples:
            ids = [r[2] for r in selected]
            reference_found = False
            for peer in path.parent.parent.glob(
                f"rank*/{path.name}/**/op_summary*.csv"
            ):
                reference = sorted(read_records(peer)[key])
                if (
                    expected_samples
                    <= len(reference)
                    <= expected_samples + warmup_samples
                ):
                    if ids == [r[2] for r in reference[-used_samples:]] and all(
                        i.isdigit() for i in ids
                    ):
                        reference_found = True
                        break
            if not reference_found:
                raise RuntimeError(
                    f"{phase}/{key}: missing records without a matching complete-rank tail"
                )
        # 在当前串行路径，稳定阶段 Task ID（或 HCCL 迭代序号）应等步长增长。
        ids = []
        for r in selected:
            if r[2].isdigit():
                ids.append(int(r[2]))
            elif key == "hccl":
                try:
                    ids.append(int(r[3].split("_")[-2]))
                except (ValueError, IndexError):
                    raise RuntimeError(
                        "Cannot identify HCCL iteration sequence"
                    ) from None
            else:
                raise RuntimeError(f"Cannot validate {phase}/{key} task sequence")
        steps = [b - a for a, b in zip(ids, ids[1:])]
        if steps and (steps[0] <= 0 or len(set(steps)) != 1):
            raise RuntimeError(f"Non-contiguous {phase}/{key} averaging window")
        samples = [r[1] for r in selected]
        result[key + "_raw_samples"] = len(records)
        result[key + "_used_samples"] = len(samples)
        result[key + "_us"] = sum(samples) / len(samples)
    if phase == "baseline":
        result["hccl_device_us"] = result["hccl_us"]
        # HCCL 启动任务未必与 RS 调用一一对应，不能套用 MM/RS 的轮数校验。
        # 未建立逐轮对应关系时不报告平均启动耗时；原始记录保留在 Profiler 中。
        result["aicpu_raw_samples"] = len(valid[0]["aicpu"])
        result["aicpu_us"] = ""
        result["aicpu_status"] = (
            "NOT_ATTRIBUTED" if valid[0]["aicpu"] else "NOT_CAPTURED"
        )
    return result


# ------------------------------ 多卡执行与结果汇总 ------------------------------
def worker(rank, args, cases, precision_inputs=None):
    """每个 worker 占用一张可见卡，执行所选 case；精度每组新建进程，性能跨组复用通信组。

    每组保存结果；异常保存 traceback 并交给父进程传播，父进程不会误判成功。
    """
    print(
        f"[ENV] torch={torch.__version__} torch_npu={torch_npu.__version__} "
        f"extension={module.__file__}",
        flush=True,
    )
    torch.npu.set_device(rank)
    if torch.npu.device_count() < args.world_size:
        raise RuntimeError("world-size exceeds visible NPU devices")
    group = None
    records = []
    current_case = None
    current_phase = "initialization"
    try:
        options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
        options.hccl_config = args.hccl_config
        print(f"[ENV] default_group_hccl_config={options.hccl_config}", flush=True)
        dist.init_process_group(
            "hccl",
            rank=rank,
            world_size=args.world_size,
            init_method=f"tcp://{args.master_ip}:{args.port}",
            pg_options=options,
        )
        group = dist.new_group(backend="hccl", ranks=list(range(args.world_size)))
        # 原 profile：同一 K/N 的 B 和 B scale 只分配一次，跨 shape 复用。
        if args.mode == "perf":
            current_phase = "input_setup"
            # 保留上一组最后一次输出，直到下一次算子调用返回后覆盖，沿用原 profile。
            output_state = [None]
            perf_weights = {}
            baseline_weights = {}
            baseline_scales = {}
            base_args = baseline_config(args) if args.with_baseline else args
            for _, _, k, n in cases:
                if (k, n) not in perf_weights:
                    b_cpu = torch.randn((n, k) if args.x2_orient == "trans" else (k, n))
                    b = build_b(quantize_cpu(b_cpu, args.x2_dtype), args)
                    sb = torch.full(
                        (n, math.ceil(k / 64), 2)
                        if args.x2_orient == "trans"
                        else (math.ceil(k / 64), n, 2),
                        127,
                        dtype=torch.uint8,
                    ).npu()
                    perf_weights[k, n] = (b, sb)
                    if args.with_baseline:
                        baseline_weights[k, n] = (
                            b
                            if base_args.x2_dtype == args.x2_dtype
                            else build_b(
                                quantize_cpu(b_cpu, base_args.x2_dtype), base_args
                            )
                        )
                        if args.x2_format == "nz" and args.x2_orient == "trans":
                            # The standalone matmul infers transpose from the B view.
                            baseline_weights[k, n] = baseline_weights[k, n].t()
                        baseline_scales[k, n] = prepare_baseline_inputs(
                            (None, b, None, sb), args
                        )[3]
            cache_input = torch.rand((49152, 1), dtype=torch.float32).npu()
            heavy = (
                torch.ones(10000, 10000).to(torch.float16).npu()
                if args.exp_count
                else None
            )
        for case in cases:
            name, m, k, n = case
            current_case = name
            current_phase = "input_setup"
            print(
                f"[rank {rank}] {name}: perRankM={m} inputM={m * args.world_size} K={k} N={n}",
                flush=True,
            )
            if args.mode == "precision":
                a, b, sa, sb = precision_inputs[rank]
                a_npu = to_npu_quant(a.clone(), args.x1_dtype)
                b_npu = build_b(b.clone(), args)
                inputs = (a_npu, b_npu, sa.clone().npu(), sb.clone().npu())
                current_phase = "precision"
                for iteration in range(args.iters):
                    output = fused(inputs, group, args)
                torch.npu.synchronize()
                dist.barrier(group=group)
                # 原 acc：子进程只执行并返回输出，父进程负责精度比对。
                result_dir = case_detail_dir(args, name)
                result_dir.mkdir(parents=True, exist_ok=True)
                result_path = result_dir / f"rank{rank}.npy"
                if output.dtype != OUTPUT_DTYPES[args.y_dtype][0]:
                    raise AssertionError(f"unexpected output dtype: {output.dtype}")
                np.save(result_path, output.cpu().float().numpy())
                records.append(
                    dict(
                        case=name,
                        rank=rank,
                        iteration=args.iters,
                        output_path=str(result_path),
                        result="PENDING",
                    )
                )
            else:
                # 每组只生成 A 和 A scale；性能模式不需要重新生成 B 或 CPU golden。
                b, sb = perf_weights[k, n]
                a_cpu = torch.randn(m * args.world_size, k)
                inputs = (
                    to_npu_quant(
                        quantize_cpu(a_cpu, args.x1_dtype),
                        args.x1_dtype,
                    ),
                    b,
                    torch.full(
                        (m * args.world_size, math.ceil(k / 64), 2),
                        127,
                        dtype=torch.uint8,
                    ).npu(),
                    sb,
                )
                path = case_detail_dir(args, name) / f"rank{rank}"
                current_phase = "fused"
                profile_calls(
                    lambda: fused(inputs, group, args),
                    path / "mc2",
                    args,
                    cache_input,
                    heavy,
                    output_state,
                )
                current_phase = "fused_barrier"
                dist.barrier(group=group)
                if args.with_baseline:
                    current_phase = "baseline"
                    baseline_out = torch.empty(
                        (m, n),
                        dtype=OUTPUT_DTYPES[args.y_dtype][0],
                        device=f"npu:{rank}",
                    )
                    baseline_inputs = (
                        inputs[0]
                        if base_args.x1_dtype == args.x1_dtype
                        else to_npu_quant(
                            quantize_cpu(a_cpu, base_args.x1_dtype), base_args.x1_dtype
                        ),
                        baseline_weights[k, n],
                        inputs[2],
                        baseline_scales[k, n],
                    )
                    profile_calls(
                        lambda: baseline(
                            baseline_inputs, group, baseline_out, base_args
                        ),
                        path / "baseline",
                        args,
                        cache_input,
                        heavy,
                        output_state,
                    )
                    # 所有 rank 完成本组 baseline 后，再进入下一组 shape。
                    current_phase = "baseline_barrier"
                    dist.barrier(group=group)
                    del baseline_inputs
                records.append(
                    dict(
                        case=name,
                        rank=rank,
                        result="DONE",
                        profile_path=str(path),
                    )
                )
            (detail_root(args) / "_workers" / f"rank{rank}.json").write_text(
                json.dumps(records, indent=2), encoding="utf-8"
            )
            del inputs
            if args.mode == "precision":
                del output
            torch.npu.empty_cache()
        (detail_root(args) / "_workers" / f"rank{rank}.json").write_text(
            json.dumps(records, indent=2), encoding="utf-8"
        )
    except BaseException:
        import traceback

        records.append(
            dict(
                case=current_case,
                rank=rank,
                result="ERROR",
                error_phase=current_phase,
                error=traceback.format_exc(),
            )
        )
        (detail_root(args) / "_workers" / f"rank{rank}.json").write_text(
            json.dumps(records, indent=2), encoding="utf-8"
        )
        (detail_root(args) / "_workers" / f"rank{rank}_error.log").write_text(
            traceback.format_exc(), encoding="utf-8"
        )
        # Do not enter communication-group teardown on the failure path: peers may
        # still be blocked inside a collective. Let the parent terminate this worker group.
        raise
    else:
        if dist.is_initialized():
            if group is not None:
                dist.destroy_process_group(group)
            dist.destroy_process_group()
            torch.npu.empty_cache()


def detail_root(args):
    if os.environ.get("QMRS_SCENARIO_DIR"):
        return args.output_dir
    return (
        args.output_dir
        / "details"
        / ("precision" if args.mode == "precision" else "performance")
    )


def case_detail_dir(args, name):
    if os.environ.get("QMRS_SCENARIO_DIR"):
        _, m, k, n = next(case for case in build_cases(args) if case[0] == name)
        return args.output_dir / f"m{m}_k{k}_n{n}"
    return detail_root(args) / name


def case_fields(args, case):
    name, m, k, n = case
    return dict(
        case=name,
        world_size=args.world_size,
        per_rank_m=m,
        input_m=m * args.world_size,
        k=k,
        n=n,
        x1_dtype=args.x1_dtype,
        x2_dtype=args.x2_dtype,
        y_dtype=args.y_dtype,
        x2_format=args.x2_format,
        x2_orient=args.x2_orient,
    )


def summarize_case(args, case, ranks):
    if len(ranks) != args.world_size or {r["rank"] for r in ranks} != set(
        range(args.world_size)
    ):
        raise RuntimeError(f"Incomplete rank results for {case[0]}")
    row = case_fields(args, case)
    if args.mode == "precision":
        row.update(
            result="PASS" if all(r["result"] == "PASS" for r in ranks) else "FAIL",
            passed_ranks=sum(r["result"] == "PASS" for r in ranks),
            max_abs_error=max(r["max_abs_error"] for r in ranks),
            rtol=args.rtol,
            atol=args.atol,
        )
        return row
    slowest = max(ranks, key=lambda r: r["fused_us"])
    base_args = baseline_config(args)
    row.update(
        mc2_us=slowest["fused_us"],
        result="DONE" if all(r["result"] == "DONE" for r in ranks) else "FAIL",
        matmul_us="",
        hccl_us="",
        baseline_us="",
        prof_rate="",
        mask_rate="",
        baseline_status="DONE" if args.with_baseline else "NOT_REQUESTED",
        baseline_x1_dtype=base_args.x1_dtype if args.with_baseline else "",
        baseline_x2_dtype=base_args.x2_dtype if args.with_baseline else "",
    )
    if args.with_baseline:
        slow_base = max(ranks, key=lambda r: r["matmul_us"] + r["hccl_us"])
        mm, hccl = slow_base["matmul_us"], slow_base["hccl_us"]
        row.update(
            baseline_us=mm + hccl,
            prof_rate=(mm + hccl) / row["mc2_us"],
            baseline_rank=slow_base["rank"],
            aicpu_us=slow_base["aicpu_us"],
            aicpu_raw_samples=slow_base["aicpu_raw_samples"],
            aicpu_status=slow_base["aicpu_status"],
            hccl_device_us=slow_base["hccl_device_us"],
            matmul_us=mm,
            hccl_us=hccl,
            mask_rate=(mm + hccl - row["mc2_us"]) / min(mm, hccl),
        )
    return row


def write_error_summary(
    args, cases, error, records=None, failed_case=None, error_phase="initialization"
):
    """仅确认过的结果写 PASS/DONE；未开始和中断的 shape 单独标记。"""
    # worker 的每组进度已在原有落盘点保存；不在性能采集循环增加 I/O。
    recovered = {}
    for filename in (detail_root(args) / "_workers").glob("rank*.json"):
        try:
            for row in json.loads(filename.read_text(encoding="utf-8")):
                if args.mode == "perf" or row["result"] == "ERROR":
                    recovered[row["case"], row["rank"]] = row
        except (OSError, ValueError, KeyError, TypeError):
            continue
    for row in records or []:
        recovered[row["case"], row["rank"]] = row
    worker_errors = [r for r in recovered.values() if r["result"] == "ERROR"]
    if worker_errors:
        failed_case = worker_errors[0].get("case") or failed_case
        error_phase = worker_errors[0].get("error_phase", error_phase)
    if failed_case is None and cases:
        failed_case = cases[0][0]
    rows = []
    for case in cases:
        ranks = [r for (name, _), r in recovered.items() if name == case[0]]
        complete = len(ranks) == args.world_size and {r["rank"] for r in ranks} == set(
            range(args.world_size)
        )
        phase = error_phase if case[0] == failed_case else ""
        if complete and all(r["result"] in ("DONE", "PASS", "FAIL") for r in ranks):
            try:
                if args.mode == "perf":
                    for rank_row in ranks:
                        path = Path(rank_row["profile_path"])
                        if "fused_us" not in rank_row:
                            phase = "fused_parse"
                            rank_row.update(
                                parse_summary(
                                    path / "mc2",
                                    "mc2",
                                    expected_samples=args.loop_time - args.warmup_time,
                                    warmup_samples=args.warmup_time,
                                )
                            )
                        if args.with_baseline and "matmul_us" not in rank_row:
                            phase = "baseline_parse"
                            rank_row.update(
                                parse_summary(
                                    path / "baseline",
                                    "baseline",
                                    expected_samples=args.loop_time - args.warmup_time,
                                    warmup_samples=args.warmup_time,
                                )
                            )
                rows.append(summarize_case(args, case, ranks))
                continue
            except (RuntimeError, OSError, ValueError, KeyError) as parse_error:
                case_error = str(parse_error)
                result = "ERROR"
        else:
            result = (
                "ERROR"
                if case[0] == failed_case
                else ("INCOMPLETE" if ranks else "NOT_RUN")
            )
            case_error = str(error) if result != "NOT_RUN" else ""
        row = case_fields(args, case)
        row.update(result=result, error=case_error, error_phase=phase)
        if args.mode == "precision":
            row.update(
                passed_ranks="", max_abs_error="", rtol=args.rtol, atol=args.atol
            )
        else:
            baseline_status = "NOT_RUN"
            if phase in ("baseline", "baseline_parse"):
                baseline_status = (
                    "PARSE_FAILED" if phase == "baseline_parse" else "FAILED"
                )
            elif phase == "baseline_barrier":
                baseline_status = "INCOMPLETE"
            row.update(
                mc2_us="",
                matmul_us="",
                hccl_us="",
                baseline_us="",
                prof_rate="",
                mask_rate="",
                baseline_status=baseline_status
                if args.with_baseline
                else "NOT_REQUESTED",
            )
            # 融合已成功解析而 baseline 失败时，保留已验证的融合耗时。
            if complete and all("fused_us" in r for r in ranks):
                row["mc2_us"] = max(r["fused_us"] for r in ranks)
        rows.append(row)
    write_csv(
        args.output_dir
        / (
            "precision_result.csv"
            if args.mode == "precision"
            else "performance_result.csv"
        ),
        rows,
    )


def write_csv(path, rows):
    batch_dir = os.environ.get("QMRS_BATCH_DIR")
    if not batch_dir:
        write_csv_file(path, rows)
        return

    root = Path(batch_dir)
    summary_path = root / path.name
    updates = {
        (
            path.parent.relative_to(root)
            / f"m{row['per_rank_m']}_k{row['k']}_n{row['n']}"
        ).as_posix(): row
        for row in rows
    }
    with summary_path.open(newline="", encoding="utf-8") as stream:
        batch_rows = list(csv.DictReader(stream))
    for row in batch_rows:
        row.update(updates.get(row["detail_dir"], {}))
    pending_path = summary_path.with_suffix(".tmp")
    write_csv_file(pending_path, batch_rows)
    pending_path.replace(summary_path)


def write_csv_file(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


SCENARIO_FIELDS = ("x1_dtype", "x2_dtype", "y_dtype", "x2_format", "x2_orient")
SCENARIO_OPTIONS = tuple("--" + field.replace("_", "-") for field in SCENARIO_FIELDS)


def run_batch(args, argv, matrix):
    phase_args = []
    tokens = iter(argv)
    for arg in tokens:
        option = arg.split("=", 1)[0]
        if option in ("--mode", "--output-dir", *SCENARIO_OPTIONS):
            if "=" not in arg:
                next(tokens)
        elif arg != "--with-baseline":
            phase_args.append(arg)

    fp8 = ("e4m3", "e5m2")
    outputs = ("bf16", "fp16")
    directions = ("trans", "non-trans")
    scenarios = (
        list(product(fp8, fp8, outputs, ("nd", "nz"), directions))
        + list(product(("fp4",), ("fp4",), outputs, ("nd",), directions))
        if matrix
        else [tuple(getattr(args, field) for field in SCENARIO_FIELDS)]
    )
    phases = ("precision", "perf") if args.mode == "all" else (args.mode,)
    cases = build_cases(args)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    batch_dir = (
        args.output_dir / f"{stamp}_{args.world_size}card_{args.mode}_{os.getpid()}"
    ).resolve()
    batch_dir.mkdir(parents=True)
    (batch_dir / "config.json").write_text(
        json.dumps(
            dict(
                vars(args),
                output_dir=str(batch_dir),
                cases=cases,
                scenarios=[dict(zip(SCENARIO_FIELDS, values)) for values in scenarios],
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    plans = {}
    for phase in phases:
        folder = "precision" if phase == "precision" else "performance"
        rows = []
        for values in scenarios:
            scenario_args = argparse.Namespace(
                **(vars(args) | dict(zip(SCENARIO_FIELDS, values)))
            )
            for case in cases:
                _, m, k, n = case
                rows.append(
                    dict(
                        case_fields(scenario_args, case),
                        result="NOT_RUN",
                        detail_dir=f"{folder}/{'_'.join(values)}/m{m}_k{k}_n{n}",
                    )
                )
        summary = batch_dir / f"{folder}_result.csv"
        write_csv_file(summary, rows)
        plans[phase] = (folder, summary)

    print(f"[BATCH] output={batch_dir}", flush=True)
    launcher = Path(__file__).resolve().with_name("run_test.sh")
    failed = False
    for phase, (folder, summary) in plans.items():
        passed = 0
        for values in scenarios:
            scenario = "_".join(values)
            location = batch_dir / folder / scenario
            options = [f"{key}={value}" for key, value in zip(SCENARIO_OPTIONS, values)]
            command = [
                "bash",
                str(launcher),
                *phase_args,
                *options,
                "--mode",
                phase,
                "--output-dir",
                str(location),
            ]
            if phase == "perf" and (args.with_baseline or args.mode == "all"):
                command.append("--with-baseline")
            env = dict(
                os.environ,
                QMRS_BATCH_ACTIVE="1",
                QMRS_BATCH_DIR=str(batch_dir),
                QMRS_SCENARIO_DIR=str(location),
            )
            print(f"[MATRIX] {phase} {scenario}", flush=True)
            rc = subprocess.run(command, env=env).returncode
            if rc:
                failed = True
                with summary.open(newline="", encoding="utf-8") as stream:
                    rows = list(csv.DictReader(stream))
                for row in rows:
                    if row["result"] == "NOT_RUN" and row["detail_dir"].startswith(
                        f"{folder}/{scenario}/"
                    ):
                        row.update(
                            result="ERROR", error=f"process exited {rc}; see run.log"
                        )
                write_csv_file(summary, rows)
            else:
                passed += 1
        print(f"[MATRIX] {phase}: {passed}/{len(scenarios)} passed", flush=True)
    return int(failed)


def main():
    args = parse_args()
    if args.list_cases:
        for name, m, k, n in DEFAULT_CASES:
            print(f"{name:8s} perRankM={m:4d} K={k} N={n}; inputM=perRankM*world-size")
        return 0
    argv = sys.argv[1:]
    matrix = not any(arg.split("=", 1)[0] in SCENARIO_OPTIONS for arg in argv)
    if args.mode == "all" or (
        os.environ.get("QMRS_MATRIX_ENTRY") == "1"
        and matrix
        and os.environ.get("QMRS_BATCH_ACTIVE") != "1"
    ):
        return run_batch(args, argv, matrix)
    cases = build_cases(args)
    if args.x1_dtype == "fp4":
        for _, _, k, n in cases:
            if k % 2 or (args.x2_orient == "non-trans" and n % 2):
                raise ValueError(
                    "FP4 requires even K; non-trans B also requires even N"
                )
    if (
        args.y_dtype == "fp16"
        and "y_dtype" not in inspect.signature(quant_matmul_reduce_scatter).parameters
    ):
        raise RuntimeError(
            "Installed extension lacks y_dtype; update operator and extension for FP16"
        )
    # A fresh directory prevents stale results from being mistaken for this run.
    run_name = time.strftime("%Y%m%d_%H%M%S") + f"_{args.mode}_{os.getpid()}"
    scenario_dir = os.environ.get("QMRS_SCENARIO_DIR")
    args.output_dir = (
        Path(scenario_dir) if scenario_dir else args.output_dir.resolve() / run_name
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (detail_root(args) / "_workers").mkdir(parents=True)
    if not scenario_dir:
        (args.output_dir / "config.json").write_text(
            json.dumps(
                {**vars(args), "output_dir": str(args.output_dir), "cases": cases},
                indent=2,
            ),
            encoding="utf-8",
        )
    print(
        f"[INFO] mode={args.mode} x2_format={args.x2_format} world_size={args.world_size} output={args.output_dir}",
        flush=True,
    )
    records = []
    failed_case = None
    error_phase = "initialization"
    try:
        # 原 acc 每个 shape 单独起进程/通信组；profile 跨 shape 复用通信组。
        batches = [[case] for case in cases] if args.mode == "precision" else [cases]
        base_port = args.port
        for index, batch in enumerate(batches):
            failed_case = batch[0][0]
            error_phase = "input_setup"
            precision_inputs, goldens = (
                prepare_precision_case(batch[0], args.world_size, args)
                if args.mode == "precision"
                else (None, None)
            )
            args.port = base_port + index if args.mode == "precision" else base_port
            if args.port >= 65536:
                raise ValueError("case port exceeds 65535")
            error_phase = "worker_execution"
            mp.start_processes(
                worker,
                args=(args, batch, precision_inputs),
                nprocs=args.world_size,
                join=True,
                start_method="forkserver",
            )
            for rank in range(args.world_size):
                rows = json.loads(
                    (detail_root(args) / "_workers" / f"rank{rank}.json").read_text(
                        encoding="utf-8"
                    )
                )
                if len(rows) != len(batch) or any(row["rank"] != rank for row in rows):
                    raise RuntimeError(f"Incomplete results for rank {rank}")
                if args.mode == "precision":
                    for row in rows:
                        failed_case = row["case"]
                        error_phase = "precision_compare"
                        actual = torch.from_numpy(np.load(row["output_path"]))
                        passed, error, close, total, npu_value, golden_value = compare(
                            actual, goldens[rank], args
                        )
                        row.update(
                            result="PASS" if passed else "FAIL", max_abs_error=error
                        )
                        print(
                            f"[rank {rank}] {row['case']}: {row['result']} "
                            f"close={close}/{total} ({100 * close / total:.2f}%) "
                            f"max_err={error:.6f} npu={npu_value:.4f} "
                            f"golden={golden_value:.4f}",
                            flush=True,
                        )
                    (
                        case_detail_dir(args, batch[0][0]) / f"rank{rank}.json"
                    ).write_text(json.dumps(rows, indent=2), encoding="utf-8")
                records.extend(rows)
        # 所有 NPU 执行完成后，父进程才解析性能 CSV，不在 shape 之间插入解析。
        if args.mode == "perf":
            for row in records:
                failed_case = row["case"]
                error_phase = "fused_parse"
                path = Path(row["profile_path"])
                row.update(
                    parse_summary(
                        path / "mc2",
                        "mc2",
                        expected_samples=args.loop_time - args.warmup_time,
                        warmup_samples=args.warmup_time,
                    )
                )
                if args.with_baseline:
                    error_phase = "baseline_parse"
                    row.update(
                        parse_summary(
                            path / "baseline",
                            "baseline",
                            expected_samples=args.loop_time - args.warmup_time,
                            warmup_samples=args.warmup_time,
                        )
                    )
        # 逐卡结果只保存 JSON；顶层 CSV 均为每用例一行。
        for row in records:
            target = case_detail_dir(args, row["case"]) / f"rank{row['rank']}.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(row, indent=2), encoding="utf-8")
        summary = [
            summarize_case(args, case, [r for r in records if r["case"] == case[0]])
            for case in cases
        ]
        write_csv(
            args.output_dir
            / (
                "precision_result.csv"
                if args.mode == "precision"
                else "performance_result.csv"
            ),
            summary,
        )
        if args.mode == "perf":
            for row in summary:
                print(f"[PERF] {row}", flush=True)
        expected_result = "PASS" if args.mode == "precision" else "DONE"
        passed = all(row["result"] == expected_result for row in records)
        print(
            f"[RESULT] {sum(row['result'] == expected_result for row in records)}/{len(records)} {expected_result} -> "
            f"{'COMPLETED' if passed else 'FAILED'}",
            flush=True,
        )
        return 0 if passed else 1
    except BaseException as error:
        write_error_summary(args, cases, error, records, failed_case, error_phase)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
